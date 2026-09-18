from dataclasses import dataclass
from typing import Callable

from .models import SearchResult

__all__ = ["EvalCase", "EvalReport", "evaluate_retriever"]


@dataclass
class EvalCase:
    """One retrieval evaluation case: a query and the document it should surface (see
    Day 12 plan item 16)."""

    query: str
    expected_document_id: str


@dataclass
class EvalReport:
    """Top-1/Top-3/Top-5 accuracy for one retriever over a set of `EvalCase`s (see Day
    12 plan item 16), each as a 0-100 percentage."""

    name: str
    top1: float
    top3: float
    top5: float
    cases: int


def evaluate_retriever(
    name: str, cases: list[EvalCase], search_fn: Callable[[str, int], list[SearchResult]]
) -> EvalReport:
    """Runs every case through `search_fn(query, limit=5)` and checks whether
    `expected_document_id` appears in the top-1/3/5 results, per Day 12 plan item 16.
    `search_fn` is a thin wrapper so the same harness works for `VectorRetriever`,
    `KeywordRetriever`, and `HybridRetriever` (or `KnowledgeService.search`)."""
    if not cases:
        return EvalReport(name=name, top1=0.0, top3=0.0, top5=0.0, cases=0)

    hits = {1: 0, 3: 0, 5: 0}
    for case in cases:
        results = search_fn(case.query, 5)
        document_ids = [result.document_id for result in results]
        for depth in hits:
            if case.expected_document_id in document_ids[:depth]:
                hits[depth] += 1

    total = len(cases)
    return EvalReport(
        name=name,
        top1=100 * hits[1] / total,
        top3=100 * hits[3] / total,
        top5=100 * hits[5] / total,
        cases=total,
    )
