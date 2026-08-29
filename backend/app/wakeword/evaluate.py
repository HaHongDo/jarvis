"""Manual wake-word evaluation harness.

Run this while performing the Day 4 test plan: repeated intentional
activations ("Hey Jarvis" x N), similar-sounding phrases ("Hey Jason",
"Okay Jarvis", ...), and background audio (TV, music, conversation,
silence). Every detection is logged with a timestamp and score so you can
correlate it against what you actually said, then tally true detections,
missed detections, and false activations by hand.

Usage:
    python -m app.wakeword.evaluate
"""

import logging
import time

from ..audio import MicrophoneStream
from .openwakeword import OpenWakeWordDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run() -> None:
    detector = OpenWakeWordDetector()
    detections = 0

    print("Listening for the wake word. Press Ctrl+C to stop and see a summary.")
    print("Try: repeated 'Hey Jarvis', similar phrases (Hey Jason/Travis, Okay/Hi Jarvis),")
    print("and background audio (TV, music, conversation, silence).")
    print()

    start = time.perf_counter()
    try:
        with MicrophoneStream() as mic:
            for frame in mic:
                if detector.process(frame):
                    detections += 1
                    elapsed = time.perf_counter() - start
                    print(f"[{elapsed:6.1f}s] detection #{detections}")
                    detector.reset()
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start
        print()
        print(f"Stopped after {elapsed:.1f}s with {detections} detection(s).")
        print("Tally these against what you actually said to compute:")
        print("  false-negative rate = missed intentional activations / intentional activations")
        print("  false-positive rate = unintended detections / total detections")


if __name__ == "__main__":
    run()
