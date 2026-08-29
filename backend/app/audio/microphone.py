import logging
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from ..config import (
    AUDIO_CHANNELS,
    AUDIO_FRAME_MS,
    AUDIO_SAMPLE_RATE,
    MAX_RECORDING_SECONDS,
    SILENCE_THRESHOLD,
    SILENCE_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)


class MicrophoneStream:
    """Continuous mono int16 PCM audio, read as fixed-size frames.

    Frame size defaults to 80ms multiples, as recommended by openWakeWord
    for low-latency, efficient processing.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        device: Optional[int] = None,
        frame_ms: int = AUDIO_FRAME_MS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._stream: Optional[sd.InputStream] = None

    def __enter__(self) -> "MicrophoneStream":
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_frame(self) -> np.ndarray:
        """Block until one frame of `frame_samples` int16 samples is available."""
        data, overflowed = self._stream.read(self.frame_samples)
        if overflowed:
            logger.warning("[MIC] input overflow: frames were dropped")
        return np.squeeze(data)

    def stream(self) -> "MicrophoneStream":
        """Alias for iterating frames: `for frame in microphone.stream(): ...`."""
        return self

    def __iter__(self) -> "MicrophoneStream":
        return self

    def __next__(self) -> np.ndarray:
        return self.read_frame()


def _rms(frame: np.ndarray) -> float:
    """Normalized (0-1) root-mean-square energy of an int16 audio frame."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768.0


class AudioRecorder:
    """Captures mono audio from the default (or a selected) input device."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device if device is not None else self._default_input_device()

    @staticmethod
    def list_devices():
        return sd.query_devices()

    @staticmethod
    def _default_input_device() -> Optional[int]:
        try:
            device = sd.default.device[0]
        except Exception:
            logger.warning("Could not determine default input device")
            return None
        return device if device != -1 else None

    def record(self, duration_seconds: float) -> np.ndarray:
        """Record up to `duration_seconds` of mono audio and return it as float32 samples."""
        logger.info("[STT] recording started (max %.1fs, device=%s)", duration_seconds, self.device)
        frames = int(duration_seconds * self.sample_rate)

        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )
        sd.wait()

        audio = np.squeeze(audio)
        logger.info("[STT] recording: %.2fs", duration_seconds)
        return audio

    def record_command(
        self,
        max_seconds: float = MAX_RECORDING_SECONDS,
        silence_timeout_ms: int = SILENCE_TIMEOUT_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
        frame_ms: int = AUDIO_FRAME_MS,
    ) -> np.ndarray:
        """Record a command after wake-word activation.

        Stops at `max_seconds`, or after `silence_timeout_ms` of silence once
        speech has been detected — whichever comes first.
        """
        silence_chunks_limit = max(1, round(silence_timeout_ms / frame_ms))
        max_chunks = max(1, round(max_seconds * 1000 / frame_ms))

        logger.info("[RECORD] command recording started (max %.1fs)", max_seconds)

        collected = []
        speech_started = False
        silence_run = 0

        with MicrophoneStream(
            sample_rate=self.sample_rate,
            channels=self.channels,
            device=self.device,
            frame_ms=frame_ms,
        ) as mic:
            for _ in range(max_chunks):
                frame = mic.read_frame()
                collected.append(frame)

                if _rms(frame) > silence_threshold:
                    speech_started = True
                    silence_run = 0
                elif speech_started:
                    silence_run += 1
                    if silence_run >= silence_chunks_limit:
                        break

        audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)
        duration = len(audio) / self.sample_rate
        logger.info("[RECORD] command recording: %.2fs", duration)

        return audio.astype(np.float32) / 32768.0

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        sf.write(path, audio, self.sample_rate)
