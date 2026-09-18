"""CLI to measure Speech Normalizer accuracy over a test corpus (Day 13 plan item
20, Day 14 items 16-17, 20):

    python -m app.speech.evaluate
    python -m app.speech.evaluate --cases path/to/terminology.json
    python -m app.speech.evaluate --by-category

Each case is a raw (simulated STT) transcript and the text it should normalize to;
an unchanged `expected` means the case is a negative test (Day 13 plan item 19).
`context_domain` pins the conversation topic for context-dependent cases, so the
same phrase can be expected to correct under one topic and survive under another
(day 14 item 17). `category` is only used to group the report.

Precision matters more than recall here: it is better to occasionally miss a
"gore teen" than to confidently rewrite an already-correct sentence.
"""

import argparse
import json
from collections import defaultdict
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
    category: str = "general"


@dataclass
class EvalReport:
    """Matches the Day 13 plan item 20 evaluation summary buckets exactly.
    `incorrectly_changed` is the false-correction count from day 14 item 20 - the
    one metric the normalizer can't self-report at runtime, because it only exists
    relative to a known-good expectation."""

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
    return [
        EvalCase(
            input=c["input"],
            expected=c["expected"],
            context_domain=c.get("context_domain"),
            category=c.get("category", "general"),
        )
        for c in raw
    ]


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


def evaluate_by_category(normalizer: SpeechNormalizer, cases: list[EvalCase]) -> dict[str, EvalReport]:
    grouped: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    return {category: evaluate(normalizer, group) for category, group in sorted(grouped.items())}


def _print_report(report: EvalReport) -> None:
    print(f"Total cases:          {report.total}")
    print()
    print(f"Corrected:            {report.corrected}")
    print(f"Correctly unchanged:  {report.correctly_unchanged}")
    print(f"Incorrectly changed:  {report.incorrectly_changed}")
    print(f"Missed corrections:   {report.missed_corrections}")
    print()
    print(f"Correction precision: {report.precision:.0f}%")
    print(f"Correction recall:    {report.recall:.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Speech Normalizer accuracy against a test corpus.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Path to a JSON test-cases file")
    parser.add_argument("--by-category", action="store_true", help="Also break the results down per domain")
    args = parser.parse_args()

    if not args.cases.exists():
        print(f"No test cases found at {args.cases}")
        return

    cases = load_cases(args.cases)
    normalizer = SpeechNormalizer()
    report = evaluate(normalizer, cases)

    print("Speech Normalizer Evaluation")
    print("============================")
    print()
    _print_report(report)

    if args.by_category:
        print()
        print("Per category")
        print("------------")
        print(f"{'category':<22}{'cases':>7}{'fixed':>7}{'kept':>7}{'false':>7}{'missed':>8}")
        # Fresh normalizer so the run-level counters printed below still describe
        # exactly one pass over the corpus.
        for category, sub in evaluate_by_category(SpeechNormalizer(), cases).items():
            print(
                f"{category:<22}{sub.total:>7}{sub.corrected:>7}{sub.correctly_unchanged:>7}"
                f"{sub.incorrectly_changed:>7}{sub.missed_corrections:>8}"
            )

    metrics = normalizer.metrics
    print()
    print("Normalizer counters (day 14 item 20)")
    print("------------------------------------")
    print(f"speech_normalizer_total:        {metrics.total}")
    print(f"speech_normalizer_corrected:    {metrics.corrected}")
    print(f"speech_normalizer_unchanged:    {metrics.unchanged}")
    print(f"speech_normalizer_uncertain:    {metrics.uncertain}")
    print(f"speech_normalizer_llm_fallback: {metrics.llm_fallback}")
    print(f"speech_normalizer_false_correction: {report.incorrectly_changed}")


if __name__ == "__main__":
    main()
