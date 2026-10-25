import os
import sys
import time
import traceback
import threading
import subprocess
import numpy as np
import sounddevice as sd
from queue import Empty, Full, Queue
from dataclasses import dataclass

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import AUDIO_SAMPLE_RATE

# The realtime model works at 48 kHz natively, so a device that can run at
# that rate is preferred: no resampling is needed between device and model.
PREFERRED_SAMPLE_RATE = AUDIO_SAMPLE_RATE

# Sample rates probed in addition to the rate a device reports.
EXTRA_SAMPLE_RATES = (48000, 44100)

# ALSA plugin PCMs that only wrap, mix or rate-convert another device. They
# duplicate real hardware in the device list and can contend for the same
# PCM, so they are hidden. The server PCMs ("default", "sysdefault",
# "pipewire", "pulse", "jack") and every real card stay visible.
ALSA_PLUGIN_ALIASES = frozenset(
    {
        "center_lfe",
        "dsnoop",
        "dmix",
        "front",
        "hdmi",
        "hdmi_spdif",
        "iec958",
        "lavrate",
        "modem",
        "phoneline",
        "plug",
        "rear",
        "route",
        "samplerate",
        "side",
        "speex",
        "speexrate",
        "surround40",
        "surround41",
        "surround50",
        "surround51",
        "surround71",
        "tee",
        "upmix",
        "vdownmix",
    }
)

# Probing a device is not free, and the result of a probe stays valid while
# the device keeps existing. Results are cached per host API and device name:
# device indices shift when PipeWire nodes come and go, the names do not.
_probe_cache: dict = {}
_PROBE_CACHE_TTL = 120.0

# Starting a JACK stream can block forever, and the native call cannot be
# cancelled. The call is therefore made from a helper thread that can be
# given up on, instead of from the thread that owns the webui.
STREAM_START_TIMEOUT = 20.0

# How long to wait for a stream that is being torn down. Closing a stream
# blocks while the server is still running, so a grace period is needed.
STREAM_CLOSE_TIMEOUT = 10.0

# A JACK pair that deadlocks the server cannot be recovered from, so it is
# tried in a throwaway process before it is really used.
PREFLIGHT_TIMEOUT = 12.0


def queue_latest(queue, audio):
    """Keep each playback stream bounded without blocking the input callback."""
    try:
        queue.put_nowait(audio)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(audio)
        except Full:
            pass


def _start_stream(stream, timeout: float = STREAM_START_TIMEOUT) -> bool:
    """Start a stream without ever blocking the caller for good.

    PortAudio can hang inside Pa_StartStream: on a JACK server a client that
    captures and plays back through two different JACK clients deadlocks the
    server, and the call never returns. The start therefore runs in a helper
    thread and is given up on when it takes too long. A hung start leaves the
    stream unusable, so the caller is told about it.
    """
    result = {}

    def worker():
        try:
            result["error"] = sd._lib.Pa_StartStream(stream._ptr)
        except Exception as error:  # pragma: no cover - depends on the driver
            result["exception"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False
    error = result.get("error")
    if error is not None and error != 0:
        raise RuntimeError(
            f"Could not start the audio stream (PortAudio error {error})."
        )
    if "exception" in result:
        raise RuntimeError(f"Could not start the audio stream: {result['exception']}")
    return True


def _close_stream(stream) -> None:
    """Abort and close a stream, giving up if the driver refuses to let go."""
    if stream is None:
        return
    try:
        sd._lib.Pa_AbortStream(stream._ptr)
    except Exception:
        pass

    def worker():
        try:
            sd._lib.Pa_CloseStream(stream._ptr)
        except Exception:
            pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(STREAM_CLOSE_TIMEOUT)
    if thread.is_alive():
        # The close is stuck in the driver. The handle is dropped anyway:
        # PortAudio would otherwise try to close it again on exit.
        print(
            "Warning: an audio stream could not be closed and was abandoned. "
            "The audio server may need a restart."
        )


def _release_stream_handle(stream) -> None:
    """Forget a stream handle that was not closed by the driver.

    Closing a stream blocks while the audio server is stuck, and the webui
    must not freeze because of that. The handle is dropped instead, so the
    abandoned stream is no longer closed a second time on exit.
    """
    try:
        if stream._ptr != sd._ffi.NULL:
            stream._ptr = sd._ffi.NULL
    except Exception:
        pass


@dataclass
class ServerAudioDevice:
    index: int = 0
    name: str = ""
    host_api: str = ""
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: int = 0
    # Channel counts the device really accepts, and the rates it accepts.
    input_channels: int = 0
    output_channels: int = 0
    input_samplerates: tuple = ()
    output_samplerates: tuple = ()

    @property
    def label(self) -> str:
        """Identity of a device, independent of its PortAudio index."""
        return f"{self.name} ({self.host_api})"


def _is_hidden_plugin(name: str, host_api: str) -> bool:
    """True for ALSA plugin PCMs that only wrap another device."""
    if "ALSA" not in host_api:
        return False
    return name.strip().lower() in ALSA_PLUGIN_ALIASES


def _format_supported(
    index: int, channels: int, samplerate: int, is_input: bool
) -> bool:
    """Ask PortAudio whether a device accepts the given format."""
    try:
        if is_input:
            sd.check_input_settings(
                device=index,
                channels=channels,
                dtype="float32",
                samplerate=samplerate,
            )
        else:
            sd.check_output_settings(
                device=index,
                channels=channels,
                dtype="float32",
                samplerate=samplerate,
            )
        return True
    except Exception:
        return False


def _rate_candidates(default_samplerate: float, host_api: str) -> list:
    """Ordered sample rates to try for a device."""
    rates = []
    default_rate = int(default_samplerate) if default_samplerate else 0
    if "JACK" in host_api:
        # JACK has a single server rate; anything else is rejected outright.
        order = [default_rate, PREFERRED_SAMPLE_RATE]
    else:
        order = [PREFERRED_SAMPLE_RATE, default_rate]
    order.extend(EXTRA_SAMPLE_RATES)
    for rate in order:
        if rate and rate not in rates:
            rates.append(rate)
    return rates


def _probe_direction(
    index: int,
    is_input: bool,
    max_channels: int,
    default_samplerate: float,
    host_api: str,
) -> tuple:
    """Return (channels, supported_rates) a device really accepts."""
    if max_channels <= 0:
        return 0, ()

    channel_order = [c for c in (2, 1) if c <= max_channels] or [1]
    rates = _rate_candidates(default_samplerate, host_api)

    for channels in channel_order:
        supported = [
            r for r in rates if _format_supported(index, channels, r, is_input)
        ]
        if supported:
            return channels, tuple(supported)
    return 0, ()


def _probe_device(device, host_api: str) -> tuple:
    """Probe the usable formats of a device, with a short lived cache."""
    key = (host_api, device["name"])
    cached = _probe_cache.get(key)
    if cached is not None and (cached[0] - time.monotonic()) < _PROBE_CACHE_TTL:
        return cached[1]

    result = (
        _probe_direction(
            device["index"],
            True,
            device["max_input_channels"],
            device["default_samplerate"],
            host_api,
        ),
        _probe_direction(
            device["index"],
            False,
            device["max_output_channels"],
            device["default_samplerate"],
            host_api,
        ),
    )
    _probe_cache[key] = (time.monotonic(), result)
    return result


def clear_device_cache():
    """Forget cached device capabilities (after a PortAudio re-initialization)."""
    _probe_cache.clear()


def check_the_device(device, type: str = "input"):
    """Check whether a device can be used for the given direction."""
    host_api = ""
    try:
        hostapis = sd.query_hostapis()
        host_api = hostapis[device["hostapi"]]["name"]
    except Exception:
        pass

    if _is_hidden_plugin(device["name"], host_api):
        return False

    if type == "input":
        return device["max_input_channels"] > 0
    else:
        return device["max_output_channels"] > 0


def list_audio_device():
    """
    Function to query audio devices and host api.
    """
    try:
        audio_device_list = sd.query_devices()
    except Exception as e:
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []

    try:
        hostapis = sd.query_hostapis()
    except Exception as e:
        print("An error occurred while querying the host api:", e)
        hostapis = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the host api:", e)
        hostapis = []

    if not hostapis:
        return [], []

    audio_input_device = []
    audio_output_device = []

    for d in audio_device_list:
        try:
            host_api = hostapis[d["hostapi"]]["name"]
        except Exception:
            continue

        if _is_hidden_plugin(d["name"], host_api):
            continue

        (input_channels, input_rates), (output_channels, output_rates) = _probe_device(
            d, host_api
        )

        if input_channels <= 0 and output_channels <= 0:
            continue

        device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=host_api,
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=int(d["default_samplerate"]),
            input_channels=input_channels,
            output_channels=output_channels,
            input_samplerates=input_rates,
            output_samplerates=output_rates,
        )

        if input_channels > 0:
            audio_input_device.append(device)
        if output_channels > 0:
            audio_output_device.append(device)

    return audio_input_device, audio_output_device


def _is_jack(device) -> bool:
    """True for a device served by the JACK host API."""
    return bool(device) and "JACK" in device.host_api


def _alsa_device_name(index: int) -> str:
    """Name of an ALSA device, or an empty string for another host API."""
    try:
        for device in sd.query_devices():
            if device["index"] == index:
                host_api = sd.query_hostapis()[device["hostapi"]]["name"]
                if "ALSA" in host_api:
                    return device["name"]
                return ""
    except Exception:
        pass
    return ""


def preflight_devices(
    input_device,
    output_device,
    input_channels: int,
    output_channels: int,
    sample_rate: int,
    block_frame: int,
    timeout: float = PREFLIGHT_TIMEOUT,
):
    """Open the device pair in a throwaway process before it is really used.

    Some JACK pairs deadlock the server: PortAudio then blocks forever inside
    Pa_StartStream and the process that opened them cannot be recovered, not
    even by closing the streams. Trying the pair in a child process keeps that
    failure away from the webui, which can report the problem and stay
    usable.

    Returns None when the pair works, and an error message otherwise.
    """
    script = os.path.join(now_dir, "rvc", "realtime", "device_preflight.py")
    command = [
        sys.executable,
        script,
        str(input_device.index),
        str(output_device.index),
        str(input_channels),
        str(output_channels),
        str(sample_rate),
        str(block_frame),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=now_dir,
        )
    except Exception as error:
        return f"The audio devices could not be checked: {error}"

    try:
        try:
            output = process.communicate(timeout=timeout)[0] or ""
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return (
                "The audio devices could not be started together. A JACK device "
                "that plays back into a sink cannot capture from that sink at "
                "the same time; pick another input or output device."
            )
    finally:
        if process.poll() is None:
            process.kill()

    output = output.strip()
    if output.startswith("PREFLIGHT_OK"):
        return None
    if output.startswith("PREFLIGHT_ERROR: "):
        return output[len("PREFLIGHT_ERROR: ") :]
    return "The audio devices could not be opened."


def _as_device(device, type: str = "input"):
    """Return a ServerAudioDevice for a device object or a PortAudio index."""
    if isinstance(device, ServerAudioDevice):
        return device
    if device is None:
        return None
    try:
        index = int(device)
    except (TypeError, ValueError):
        return None
    inputs, outputs = list_audio_device()
    pool = inputs if type == "input" else outputs
    for entry in pool:
        if entry.index == index:
            return entry
    return None


def find_device(label: str, type: str = "input"):
    """Look up a device by its label, ignoring the position in the list.

    PortAudio indices are not stable: PipeWire nodes come and go, so the
    index of a JACK device moves. The label carries the name and the host API,
    which stay the same, so a saved selection survives a refresh.
    """
    if not label:
        return None
    inputs, outputs = list_audio_device()
    pool = inputs if type == "input" else outputs
    for device in pool:
        if label == device.label:
            return device
    return None


def resolve_sample_rate(
    input_device,
    asio_enabled: bool = False,
    audio_sample_rate: int = AUDIO_SAMPLE_RATE,
    output_device=None,
) -> int:
    """Return the sample rate to use for audio streaming.

    For ASIO the user-selected rate is used as-is because ASIO drivers
    do not expose a reliable default_samplerate via PortAudio.
    For all other host APIs a rate both devices accept is used, preferring the
    rate the realtime model works at natively.
    """
    if asio_enabled:
        return audio_sample_rate

    input_dev = _as_device(input_device, "input")
    if input_dev is None:
        return audio_sample_rate

    candidates = list(input_dev.input_samplerates or ())
    if output_device is not None:
        output_dev = _as_device(output_device, "output")
        if output_dev is not None and output_dev.output_samplerates:
            shared = [r for r in candidates if r in output_dev.output_samplerates]
            if shared:
                candidates = shared

    if not candidates:
        return input_dev.default_samplerate or audio_sample_rate
    if PREFERRED_SAMPLE_RATE in candidates:
        return PREFERRED_SAMPLE_RATE
    if input_dev.default_samplerate in candidates:
        return input_dev.default_samplerate
    return candidates[0]


class Audio:
    def __init__(
        self,
        callbacks,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
    ):
        self.callbacks = callbacks
        self.out_queue = Queue(maxsize=2)
        self.mon_queue = Queue(maxsize=2)
        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.monitor = None
        self.running = False
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.volume_envelope = volume_envelope
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposed_pitch = proposed_pitch
        self.proposed_pitch_threshold = proposed_pitch_threshold

    def get_input_audio_device(self, index: int):
        return _as_device(index, "input")

    def get_output_audio_device(self, index: int):
        return _as_device(index, "output")

    def process_data(self, indata: np.ndarray):
        indata = indata * self.input_audio_gain
        unpacked_data = _mix_to_mono(indata)

        return self.callbacks.change_voice(
            unpacked_data,
            self.f0_up_key,
            self.index_rate,
            self.protect,
            self.volume_envelope,
            self.f0_autotune,
            self.f0_autotune_strength,
            self.proposed_pitch,
            self.proposed_pitch_threshold,
        )

    def process_data_with_time(self, indata: np.ndarray):
        out_wav, vol, performance_ms = self.process_data(indata)
        # print(f"real-time voice conversion performance: {performance_ms:.2f} ms")
        self.latency = performance_ms  # latency to display on the application interface
        self.volume = vol

        return out_wav

    def audio_stream_no_output_callback(
        self, indata: np.ndarray, frames, times, status
    ):
        if not self.running:
            return
        try:
            out_wav = self.process_data_with_time(indata)

            queue_latest(self.out_queue, out_wav)
            if self.use_monitor:
                queue_latest(self.mon_queue, out_wav)
        except Exception as error:
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())

    def audio_stream_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        if not self.running:
            outdata.fill(0)
            return
        try:
            out_wav = self.process_data_with_time(indata)

            output_channels = outdata.shape[1]
            if self.use_monitor:
                queue_latest(self.mon_queue, out_wav)

            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())

    def audio_queue(self, q: Queue, outdata: np.ndarray, gain: float):
        if not self.running:
            outdata.fill(0)
            return
        try:
            try:
                mon_wav = q.get_nowait()
                while True:
                    try:
                        latest = q.get_nowait()
                    except Empty:
                        break
                    mon_wav = latest
            except Empty:
                outdata.fill(0)
                return

            output_channels = outdata.shape[1]
            outdata[:] = (
                np.repeat(mon_wav, output_channels).reshape(-1, output_channels) * gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio queue: {error}")
            print(traceback.format_exc())

    def run_audio_stream(
        self,
        block_frame: int,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int,
        input_max_channel: int,
        output_max_channel: int,
        output_monitor_max_channel: int,
        input_extra_setting,
        output_extra_setting,
        output_monitor_extra_setting,
        audio_sample_rate: int,
    ):
        use_asio = isinstance(input_extra_setting, sd.AsioSettings) or isinstance(
            output_extra_setting, sd.AsioSettings
        )
        # A duplex stream on a single ALSA PCM can crash the ALSA backend, so
        # the two directions are opened separately there.
        use_duplex_stream = not (
            input_device_id == output_device_id and _alsa_device_name(input_device_id)
        )
        if use_asio:
            # Re-initialize PortAudio to ensure a clean ASIO driver state.
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        if input_device_id != output_device_id:
            # Only use in cases involving different backends devices.
            self.input_stream = sd.InputStream(
                callback=self.audio_stream_no_output_callback,
                latency="low",
                dtype=np.float32,
                device=input_device_id,
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=input_max_channel,
                extra_settings=input_extra_setting,
            )
            self.output_stream = sd.OutputStream(
                callback=lambda outdata, frames, times, status: self.audio_queue(
                    self.out_queue, outdata, self.output_audio_gain
                ),
                latency="low",
                dtype=np.float32,
                device=output_device_id,
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=output_max_channel,
                extra_settings=output_extra_setting,
            )
            if not _start_stream(self.input_stream):
                raise RuntimeError(
                    "The input device did not start. It is most likely already "
                    "in use by another application."
                )
            if not _start_stream(self.output_stream):
                raise RuntimeError(
                    "The output device did not start. It is most likely already "
                    "in use by another application."
                )
        else:
            # A single duplex stream is cheaper, but not every device pair
            # supports it: a duplex stream on a single ALSA PCM crashes the
            # ALSA backend. The separate streams work everywhere, so they are
            # used for the host APIs that are known to misbehave.
            if not use_duplex_stream:
                self.input_stream = sd.InputStream(
                    callback=self.audio_stream_no_output_callback,
                    latency="low",
                    dtype=np.float32,
                    device=input_device_id,
                    blocksize=block_frame,
                    samplerate=audio_sample_rate,
                    channels=input_max_channel,
                    extra_settings=input_extra_setting,
                )
                self.output_stream = sd.OutputStream(
                    callback=lambda outdata, frames, times, status: self.audio_queue(
                        self.out_queue, outdata, self.output_audio_gain
                    ),
                    latency="low",
                    dtype=np.float32,
                    device=output_device_id,
                    blocksize=block_frame,
                    samplerate=audio_sample_rate,
                    channels=output_max_channel,
                    extra_settings=output_extra_setting,
                )
                if not _start_stream(self.input_stream):
                    raise RuntimeError(
                        "The input device did not start. It is most likely already "
                        "in use by another application."
                    )
                if not _start_stream(self.output_stream):
                    raise RuntimeError(
                        "The output device did not start. It is most likely already "
                        "in use by another application."
                    )
            else:
                self.stream = sd.Stream(
                    callback=self.audio_stream_callback,
                    latency="low",
                    dtype=np.float32,
                    device=(input_device_id, output_device_id),
                    blocksize=block_frame,
                    samplerate=audio_sample_rate,
                    channels=(input_max_channel, output_max_channel),
                    extra_settings=(input_extra_setting, output_extra_setting),
                )
                if not _start_stream(self.stream):
                    raise RuntimeError(
                        "The audio device did not start. It is most likely already "
                        "in use by another application."
                    )

        if self.use_monitor:
            self.monitor = sd.OutputStream(
                callback=lambda outdata, frames, times, status: self.audio_queue(
                    self.mon_queue, outdata, self.monitor_audio_gain
                ),
                latency="low",
                dtype=np.float32,
                device=output_monitor_id,
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=output_monitor_max_channel,
                extra_settings=output_monitor_extra_setting,
            )
            if not _start_stream(self.monitor):
                raise RuntimeError("The monitor device did not start.")

    def stop(self):
        self.running = False

        streams = [self.stream, self.input_stream, self.output_stream, self.monitor]
        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.monitor = None

        for s in streams:
            if s is not None:
                _close_stream(s)
                _release_stream_handle(s)

        while not self.out_queue.empty():
            try:
                self.out_queue.get_nowait()
            except Exception:
                break
        while not self.mon_queue.empty():
            try:
                self.mon_queue.get_nowait()
            except Exception:
                break

    def start(
        self,
        input_device,
        output_device,
        output_monitor_device=None,
        exclusive_mode: bool = False,
        asio_input_channel: int = 0,
        asio_output_channel: int = 0,
        asio_output_monitor_channel: int = 0,
        block_frame: int = 24576,
        audio_sample_rate: int = AUDIO_SAMPLE_RATE,
        asio_output_stereo: bool = True,
    ):
        self.stop()

        input_audio_device = _as_device(input_device, "input")
        output_audio_device = _as_device(output_device, "output")
        if input_audio_device is None or output_audio_device is None:
            raise RuntimeError("The selected audio device is not available.")

        input_channels = (
            input_audio_device.input_channels or input_audio_device.max_input_channels
        )
        output_channels = (
            output_audio_device.output_channels
            or output_audio_device.max_output_channels
        )

        # A JACK pair that deadlocks the server cannot be recovered from, so
        # it is tested in a throwaway process first.
        if _is_jack(input_audio_device) and _is_jack(output_audio_device):
            problem = preflight_devices(
                input_audio_device,
                output_audio_device,
                input_channels,
                output_channels,
                audio_sample_rate,
                block_frame,
            )
            if problem:
                raise RuntimeError(problem)

        (
            input_extra_setting,
            output_extra_setting,
            output_monitor_extra_setting,
            monitor_channels,
        ) = (None, None, None, None)

        if "WASAPI" in input_audio_device.host_api:
            input_extra_setting = sd.WasapiSettings(
                exclusive=exclusive_mode, auto_convert=not exclusive_mode
            )
        elif "ASIO" in input_audio_device.host_api and asio_input_channel != -1:
            input_extra_setting = sd.AsioSettings(
                channel_selectors=[asio_input_channel]
            )
            input_channels = 1

        if "WASAPI" in output_audio_device.host_api:
            output_extra_setting = sd.WasapiSettings(
                exclusive=exclusive_mode, auto_convert=not exclusive_mode
            )
        elif "ASIO" in output_audio_device.host_api and asio_output_channel != -1:
            output_selectors = (
                [asio_output_channel, asio_output_channel + 1]
                if asio_output_stereo
                else [asio_output_channel]
            )
            output_extra_setting = sd.AsioSettings(channel_selectors=output_selectors)
            output_channels = len(output_selectors)

        output_monitor_id = None
        if self.use_monitor and output_monitor_device is not None:
            output_monitor_device = _as_device(output_monitor_device, "output")
            if output_monitor_device is None:
                raise RuntimeError("The selected monitor device is not available.")
            monitor_channels = (
                output_monitor_device.output_channels
                or output_monitor_device.max_output_channels
            )
            output_monitor_id = output_monitor_device.index

            if "WASAPI" in output_monitor_device.host_api:
                output_monitor_extra_setting = sd.WasapiSettings(
                    exclusive=exclusive_mode,
                    auto_convert=not exclusive_mode,
                )
            elif (
                "ASIO" in output_monitor_device.host_api
                and asio_output_monitor_channel != -1
            ):
                output_monitor_extra_setting = sd.AsioSettings(
                    channel_selectors=[asio_output_monitor_channel]
                )
                monitor_channels = 1

        try:
            self.run_audio_stream(
                block_frame,
                input_audio_device.index,
                output_audio_device.index,
                output_monitor_id,
                input_channels,
                output_channels,
                monitor_channels,
                input_extra_setting,
                output_extra_setting,
                output_monitor_extra_setting,
                audio_sample_rate,
            )
            self.running = True
        except Exception as error:
            print(f"An error occurred while streaming audio: {error}")
            print(traceback.format_exc())
            # A half started stream would keep the device busy.
            self.stop()
            raise


def _mix_to_mono(indata: np.ndarray) -> np.ndarray:
    """Average the channels of an input block into a single mono signal."""
    if indata.ndim == 1:
        return indata
    channels = indata.T
    if channels.shape[0] <= 1:
        return channels.reshape(-1)
    # Drivers pad a mono source with a silent second channel. Averaging that
    # in would halve the level, so silent channels are dropped first.
    levels = np.sqrt(np.mean(np.square(channels, dtype=np.float32), axis=1))
    keep = levels > 1e-6
    if not keep.all():
        if not keep.any():
            keep = np.ones_like(keep)
        channels = channels[keep]
    return channels.mean(axis=0)
