"""Try opening a pair of audio devices in a separate process.

A JACK client that captures from one JACK device and plays back to another
one can deadlock the JACK server: PortAudio then blocks forever inside
Pa_StartStream and the process cannot be recovered, not even by closing the
streams. Trying the pair in a throwaway process keeps that failure away from
the webui, which can report the problem and stay usable.

The script is run as a child process, so it deliberately imports nothing but
numpy and sounddevice: pulling in the model stack there would cost seconds.
"""

import os
import sys

import numpy as np
import sounddevice as sd

now_dir = os.getcwd()
sys.path.append(now_dir)

OK = "PREFLIGHT_OK"
ERROR = "PREFLIGHT_ERROR: "


def _silence_callback(indata, frames, times, status):
    return None


def _mute_callback(outdata, frames, times, status):
    outdata.fill(0)


def _open_streams(
    input_index,
    output_index,
    input_channels,
    output_channels,
    sample_rate,
    block_frame,
):
    """Open the streams the way the realtime pipeline opens them."""
    if input_index == output_index:
        return (
            sd.Stream(
                device=(input_index, output_index),
                channels=(input_channels, output_channels),
                samplerate=sample_rate,
                blocksize=block_frame,
                dtype=np.float32,
                latency="low",
                callback=lambda indata, outdata, frames, times, status: outdata.fill(0),
            ),
            None,
        )
    return (
        sd.InputStream(
            device=input_index,
            channels=input_channels,
            samplerate=sample_rate,
            blocksize=block_frame,
            dtype=np.float32,
            latency="low",
            callback=_silence_callback,
        ),
        sd.OutputStream(
            device=output_index,
            channels=output_channels,
            samplerate=sample_rate,
            blocksize=block_frame,
            dtype=np.float32,
            latency="low",
            callback=_mute_callback,
        ),
    )


def main(argv):
    if len(argv) != 7:
        print(ERROR + "wrong number of arguments")
        return 2
    (
        input_index,
        output_index,
        input_channels,
        output_channels,
        sample_rate,
        block_frame,
    ) = (int(value) for value in argv[1:])

    streams = []
    try:
        first, second = _open_streams(
            input_index,
            output_index,
            input_channels,
            output_channels,
            sample_rate,
            block_frame,
        )
        streams = [stream for stream in (first, second) if stream is not None]
        for stream in streams:
            stream.start()
        print(OK, flush=True)
    except Exception as error:
        print(ERROR + str(error), flush=True)
    finally:
        for stream in streams:
            try:
                stream.abort()
            except Exception:
                pass
        for stream in streams:
            try:
                stream.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
