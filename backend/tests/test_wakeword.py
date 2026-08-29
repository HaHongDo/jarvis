from pathlib import Path

import numpy as np
import pytest

from app.wakeword.base import WakeWordDetector

AUDIO_DIR = Path(__file__).parent / "audio"
FRAME_SAMPLES = 1280  # 80ms @ 16kHz


def test_wakeword_detector_is_abstract():
    with pytest.raises(TypeError):
        WakeWordDetector()


class _StubDetector(WakeWordDetector):
    """Detects a wake word whenever the frame exceeds a fixed peak amplitude."""

    def __init__(self, threshold: int = 10000):
        self.threshold = threshold

    def process(self, audio_frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(audio_frame)) > self.threshold)


def test_stub_detector_flags_loud_frame():
    detector = _StubDetector()
    loud_frame = np.full(FRAME_SAMPLES, 20000, dtype=np.int16)

    assert detector.process(loud_frame) is True


def test_stub_detector_ignores_quiet_frame():
    detector = _StubDetector()
    quiet_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)

    assert detector.process(quiet_frame) is False


def test_default_reset_is_a_noop():
    detector = _StubDetector()

    detector.reset()  # should not raise


@pytest.fixture(scope="module")
def openwakeword_detector():
    pytest.importorskip("openwakeword")
    from app.wakeword import OpenWakeWordDetector

    return OpenWakeWordDetector()


def _load_frames(name):
    sf = pytest.importorskip("soundfile")
    path = AUDIO_DIR / name
    if not path.exists():
        pytest.skip(f"missing fixture: {path}")
    audio, _sample_rate = sf.read(path, dtype="int16")
    n_frames = len(audio) // FRAME_SAMPLES
    return [audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(n_frames)]


def test_hey_jarvis_wav_triggers_detection(openwakeword_detector):
    frames = _load_frames("hey_jarvis.wav")

    detected = any(openwakeword_detector.process(frame) for frame in frames)

    assert detected


def test_empty_wav_does_not_trigger_detection(openwakeword_detector):
    frames = _load_frames("empty.wav")

    detected = any(openwakeword_detector.process(frame) for frame in frames)

    assert not detected
