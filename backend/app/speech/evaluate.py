"""CLI to measure Speech Normalizer accuracy over a test corpus (see plan item 20):

    python -m app.speech.evaluate
    python -m app.speech.evaluate --cases path/to/terminology.json

Each case is a raw (simulated STT) transcript and the text it should normalize
to; an unchanged `expected` means the case is a negative test (plan item 19).
Precision matters more than recall here: it is better to occasionally miss a
"gore teen" than to confidently rewrite an already-correct sentence.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .normalizer import SpeechNormalizer
from .vocabulary import context_for_domain

_DEFAULT_CASES_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "speech" / "terminology.json"


@dataclass
class EvalCase:
    input: str
    expected: str
    context_domain: Optional[str] = None


@dataclass
class EvalReport:
    """Matches the plan item 20 evaluation summary buckets exactly."""

    total: int
    corrected: int
    correctly_unchanged: int
    incorrectly_changed: int
    missed_corrections: int

    @property
    def precision(self) -> float:
        denom = self.corrected + self.incorrectly_changed
        return 100 * self.corrected / denom if denom else 100.0

    @property
    def recall(self) -> float:
        denom = self.corrected + self.missed_corrections
        return 100 * self.corrected / denom if denom else 100.0


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text())
    return [EvalCase(input=c["input"], expected=c["expected"], context_domain=c.get("context_domain")) for c in raw]


def evaluate(normalizer: SpeechNormalizer, cases: list[EvalCase]) -> EvalReport:
    corrected = correctly_unchanged = incorrectly_changed = missed = 0

    for case in cases:
        context = context_for_domain(case.context_domain) if case.context_domain else None
        result = normalizer.normalize(case.input, context=context)
        expects_change = case.expected != case.input

        if result.normalized == case.expected:
            if expects_change:
                corrected += 1
            else:
                correctly_unchanged += 1
        elif result.normalized == case.input:
            missed += 1
        else:
            incorrectly_changed += 1

    return EvalReport(
        total=len(cases),
        corrected=corrected,
        correctly_unchanged=correctly_unchanged,
        incorrectly_changed=incorrectly_changed,
        missed_corrections=missed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Speech Normalizer accuracy against a test corpus.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Path to a JSON test-cases file")
    args = parser.parse_args()

    if not args.cases.exists():
        print(f"No test cases found at {args.cases}")
        return

    cases = load_cases(args.cases)
    report = evaluate(SpeechNormalizer(), cases)

    print("Speech Normalizer Evaluation")
    print("============================")
    print()
    print(f"Total cases:          {report.total}")
    print()
    print(f"Corrected:            {report.corrected}")
    print(f"Correctly unchanged:  {report.correctly_unchanged}")
    print(f"Incorrectly changed:  {report.incorrectly_changed}")
    print(f"Missed corrections:   {report.missed_corrections}")
    print()
    print(f"Correction precision: {report.precision:.0f}%")
    print(f"Correction recall:    {report.recall:.0f}%")


if __name__ == "__main__":
    main()
