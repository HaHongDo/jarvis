"""CLI to measure whether feeding the active vocabulary *into* STT actually helps
(day 14 items 12-13):

    python -m app.speech.evaluate_stt
    python -m app.speech.evaluate_stt --cases tests/speech/stt_cases.json
    python -m app.speech.evaluate_stt --no-hotwords      # initial_prompt only

Every recording is transcribed four ways, so the two mechanisms can be told
apart:

    audio -> STT                          baseline
          -> STT + normalizer             Day 13
          -> STT with vocabulary hints     day 14 items 11-12
          -> STT with hints + normalizer   day 14 item 14

The point is to find out whether hints help *this* setup, not to assume they do.
Whisper's `initial_prompt` and faster-whisper's `hotwords` are both probabilistic
nudges; if the "hinted" columns don't beat the baseline here, turn them off in
config.yaml (`speech.vocabulary.stt_prompt_enabled` / `stt_hotwords_enabled`) and
keep relying on the normalizer.

Recording the corpus is a manual step - 20-30 sentences of real technical speech.
Copy `tests/speech/stt_cases.example.json` to `tests/speech/stt_cases.json`,
record its sentences into `tests/speech/audio/`, and adjust the expected terms.
Paths in the manifest are resolved relative to the manifest itself.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .matching import phrase_pattern
from .normalizer import SpeechNormalizer
from .vocabulary_manager import VocabularyManager

_SPEECH_TESTS_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "speech"
_DEFAULT_CASES_PATH = _SPEECH_TESTS_DIR / "stt_cases.json"


@dataclass
class SttCase:
    audio: Path
    terms: list[str]
    domain: Optional[str] = None
    reference: str = ""


@dataclass
class Tally:
    """How many expected terms survived transcription, and in how many clips all
    of them did."""

    terms_total: int = 0
    terms_found: int = 0
    clips_perfect: int = 0
    transcripts: list[str] = field(default_factory=list)

    def add(self, transcript: str, terms: list[str]) -> None:
        found = sum(1 for term in terms if phrase_pattern(term).search(transcript))
        self.terms_total += len(terms)
        self.terms_found += found
        if found == len(terms):
            self.clips_perfect += 1
        self.transcripts.append(transcript)

    @property
    def term_accuracy(self) -> float:
        return 100 * self.terms_found / self.terms_total if self.terms_total else 0.0


def load_cases(path: Path) -> list[SttCase]:
    raw = json.loads(path.read_text())
    cases = []
    for entry in raw:
        audio = Path(entry["audio"])
        if not audio.is_absolute():
            audio = path.parent / audio
        cases.append(
            SttCase(
                audio=audio,
                terms=entry["terms"],
                domain=entry.get("domain"),
                reference=entry.get("reference", ""),
            )
        )
    return cases


def _context_for(case: SttCase) -> VocabularyManager:
    """A manager primed with the case's topic, exactly as a real conversation
    would have left it before this utterance."""
    manager = VocabularyManager()
    if case.domain:
        manager.switch_to(case.domain)
    return manager


def run(cases: list[SttCase], stt, use_hotwords: bool = True, verbose: bool = False) -> dict[str, Tally]:
    normalizer = SpeechNormalizer()
    tallies = {name: Tally() for name in ("baseline", "baseline+norm", "hinted", "hinted+norm")}

    import soundfile as sf  # imported late so --help works without an audio stack

    for case in cases:
        if not case.audio.exists():
            print(f"skipping missing recording: {case.audio}")
            continue

        audio, _sample_rate = sf.read(case.audio, dtype="float32")
        manager = _context_for(case)
        context = manager.context

        baseline = stt.transcribe(audio)
        hinted = stt.transcribe(
            audio,
            initial_prompt=manager.stt_prompt(),
            hotwords=manager.hotwords() if use_hotwords else None,
        )

        tallies["baseline"].add(baseline, case.terms)
        tallies["baseline+norm"].add(
            normalizer.normalize(baseline, conversation_context=context).normalized, case.terms
        )
        tallies["hinted"].add(hinted, case.terms)
        tallies["hinted+norm"].add(
            normalizer.normalize(hinted, conversation_context=context).normalized, case.terms
        )

        if verbose:
            print(f"\n{case.audio.name}  [{case.domain or 'no domain'}]  expecting: {', '.join(case.terms)}")
            if case.reference:
                print(f"  reference: {case.reference}")
            print(f"  baseline:  {baseline}")
            print(f"  hinted:    {hinted}")

    return tallies


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare STT accuracy with and without vocabulary hints.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Path to a JSON manifest")
    parser.add_argument("--no-hotwords", action="store_true", help="Use initial_prompt only, no hotwords")
    parser.add_argument("--verbose", action="store_true", help="Print every transcript")
    args = parser.parse_args()

    if not args.cases.exists():
        print(f"No STT cases found at {args.cases}")
        print(f"Start from the example manifest: {_SPEECH_TESTS_DIR / 'stt_cases.example.json'}")
        return

    cases = load_cases(args.cases)
    if not cases:
        print("No cases in the manifest.")
        return

    from ..stt import FasterWhisperSTT  # imported late: loading Whisper takes a while

    tallies = run(cases, FasterWhisperSTT(), use_hotwords=not args.no_hotwords, verbose=args.verbose)

    clips = len(tallies["baseline"].transcripts)
    print()
    print("STT Vocabulary Evaluation")
    print("=========================")
    print()
    print(f"Clips: {clips}   hotwords: {'off' if args.no_hotwords else 'on'}")
    print()
    print(f"{'variant':<16}{'terms':>8}{'found':>8}{'accuracy':>11}{'clean clips':>13}")
    for name, tally in tallies.items():
        print(
            f"{name:<16}{tally.terms_total:>8}{tally.terms_found:>8}"
            f"{tally.term_accuracy:>10.0f}%{tally.clips_perfect:>13}"
        )
    print()
    print("If 'hinted' does not beat 'baseline', the vocabulary hints are not earning")
    print("their keep on this setup - disable them and keep the normalizer.")


if __name__ == "__main__":
    main()
