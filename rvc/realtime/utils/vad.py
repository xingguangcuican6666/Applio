import webrtcvad
import numpy as np
from collections import deque


class VADProcessor:
    def __init__(
        self,
        sensitivity_mode=1,
        sample_rate=16000,
        frame_duration_ms=30,
        hangover_ms=200,
        crest_limit=6.0,
        voiced_fraction=0.25,
        history_ms=400,
        min_voiced_frames=2,
        floor_margin=3.0,
        noise_seed=0.002,
    ):
        """
        Initializes the VADProcessor.

        A block counts as speech only when enough of the frames in a short
        history look voiced. A single frame is never enough, because ambient
        noise, keyboard and mouse clicks and mains hum all produce isolated
        frames that a plain energy or WebRTC decision would let through.

        Args:
            sensitivity_mode (int): VAD sensitivity (0-3). 3 is most aggressive.
            sample_rate (int): Sample rate of the audio. Must be 8000, 16000, 32000, or 48000 Hz.
                               WebRTC VAD internally works best with 16000 Hz.
            frame_duration_ms (int): Duration of each audio frame in ms. Must be 10, 20, or 30.
            hangover_ms (int): How long a block keeps counting as speech after
                               the last voiced frame.
            crest_limit (float): Peak to RMS ratio above which a frame is an impulse
                                 (a key press, a mouse click) rather than a voice.
            voiced_fraction (float): Share of the recent frames that must look voiced.
            history_ms (int): Length of the sliding window the share is measured over.
            min_voiced_frames (int): Absolute minimum of voiced frames in the window.
            floor_margin (float): How many times above the learned noise floor
                                  a block must be.
        """

        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("VAD sample rate must be 8000, 16000, 32000, or 48000 Hz")
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")

        self.vad = webrtcvad.Vad(sensitivity_mode)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.hangover_samples = int(sample_rate * hangover_ms / 1000)
        self.remaining_samples = 0
        # A quiet room sits around -54 dBFS. Starting there keeps the first
        # blocks of a session from passing the level gate before the real
        # noise floor has been learned.
        self.noise_rms = noise_seed
        self.crest_limit = float(crest_limit)
        self.voiced_fraction = float(voiced_fraction)
        self.min_voiced_frames = int(min_voiced_frames)
        self.floor_margin = float(floor_margin)
        history_frames = max(
            self.min_voiced_frames, int(history_ms / frame_duration_ms)
        )
        self.history = deque(maxlen=history_frames)
        self.window = np.hanning(self.frame_length)
        self.fft_size = 1 << (2 * self.frame_length - 1).bit_length()
        frequencies = np.fft.rfftfreq(self.fft_size, 1 / sample_rate)
        self.voice_band = (frequencies >= 70) & (frequencies <= 4000)
        self.lags = np.arange(
            max(1, sample_rate // 1100), min(self.frame_length - 1, sample_rate // 70)
        )

    def _is_voice_frame(self, frame):
        """True when a single frame looks like a voice rather than a sound effect."""
        signal = frame.astype(np.float64) / 32767
        signal -= signal.mean()
        rms = float(np.sqrt(np.mean(signal**2)))
        if rms <= 0:
            return False, rms

        # A key press, a mouse click or any other mechanical transient has a
        # peak far above its own RMS. Voiced speech and singing do not.
        crest = np.max(np.abs(signal)) / rms
        if crest > self.crest_limit:
            return False, rms

        spectrum = np.abs(np.fft.rfft(signal * self.window, self.fft_size)) ** 2
        total = spectrum.sum()
        if total <= 0 or spectrum[self.voice_band].sum() < total * 0.75:
            # Reject DC, mains hum, fans and other mostly out of band noise.
            return False, rms

        transformed = np.fft.rfft(signal, self.fft_size)
        correlation = np.fft.irfft(np.abs(transformed) ** 2, self.fft_size)
        energy = np.concatenate(([0.0], np.cumsum(signal**2)))
        denominator = np.sqrt(energy[-self.lags - 1] * (energy[-1] - energy[self.lags]))
        periodicity = correlation[self.lags] / np.maximum(denominator, 1e-12)

        # The periodicity check keeps sustained vowels and singing, which
        # conversational WebRTC VAD rejects.
        voiced = bool(
            self.vad.is_speech(frame.tobytes(), self.sample_rate)
            or periodicity.max() >= 0.8
        )
        return voiced, rms

    def is_speech(self, audio_chunk_float32):
        """
        Detects if the given audio chunk contains speech.

        Args:
            audio_chunk_float32 (np.ndarray): A chunk of audio data in float32 format, mono.
                                              The sample rate must match the one VAD was initialized with.

        Returns:
            bool: True if speech is detected in the chunk, False otherwise.
        """

        if audio_chunk_float32.ndim > 1 and audio_chunk_float32.shape[1] == 1:
            audio_chunk_float32 = audio_chunk_float32.flatten()
        elif audio_chunk_float32.ndim > 1:
            # If stereo, average to mono. This is a simple approach.
            # For better results, ensure mono input from the source.
            print("VAD Warning: Received stereo audio, averaging to mono.")
            audio_chunk_float32 = np.mean(audio_chunk_float32, axis=1)

        if audio_chunk_float32.size == 0:
            return False
        chunk_samples = len(audio_chunk_float32)
        # Convert float32 audio to int16 PCM
        # WebRTC VAD expects 16-bit linear PCM audio.
        if np.max(np.abs(audio_chunk_float32)) > 1.0:
            audio_chunk_float32 = np.clip(audio_chunk_float32, -1.0, 1.0)

        audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)

        # Analyze the last partial frame as well, so word endings are retained.
        padding_size = (-len(audio_chunk_int16)) % self.frame_length
        if padding_size:
            audio_chunk_int16 = np.pad(audio_chunk_int16, (0, padding_size))
        num_frames = len(audio_chunk_int16) // self.frame_length
        if num_frames == 0:
            return False
        try:
            frame_rms = []
            for i in range(num_frames):
                start = i * self.frame_length
                end = start + self.frame_length
                frame = audio_chunk_int16[start:end]
                voiced, rms = self._is_voice_frame(frame)
                self.history.append(voiced)
                frame_rms.append(rms)

            voiced_frames = sum(self.history)
            if voiced_frames < self.min_voiced_frames or (
                voiced_frames / len(self.history) < self.voiced_fraction
            ):
                speech = False
            else:
                level = float(np.sqrt(np.mean(np.square(frame_rms))))
                floor = max(0.0005, self.noise_rms * self.floor_margin)
                speech = level >= floor

            if speech:
                self.remaining_samples = self.hangover_samples
                return True

            keep_tail = self.remaining_samples > 0
            self.remaining_samples = max(0, self.remaining_samples - chunk_samples)
            if not keep_tail:
                # Only a quiet block teaches the noise floor, otherwise a loud
                # room would raise its own gate until nothing passes any more.
                level = float(np.sqrt(np.mean(np.square(frame_rms))))
                self.noise_rms = 0.95 * self.noise_rms + 0.05 * min(level, 0.01)
            return keep_tail
        except Exception as e:
            # webrtcvad can sometimes throw "Error talking to VAD" or similar
            # if frame length is not perfect.
            print(
                f"VAD processing error: {e}. Chunk length: {len(audio_chunk_int16)}, Frame length: {self.frame_length}"
            )
            # Fallback: assume no speech on error to avoid processing noise
            return False
