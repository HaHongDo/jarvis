"""CLI to compare vector/keyword/hybrid retrieval accuracy over a set of eval cases
(see Day 12 plan item 16):

    python -m app.knowledge.evaluate
    python -m app.knowledge.evaluate --cases path/to/cases.json

Each case is a query plus the document id (its path relative to the knowledge
directory) it should surface. Add cases to `eval_cases.json` as documents are added to
the knowledge base - a retriever with no cases covering it can't be scored.
"""

import argparse
import json
from pathlib import Path

from ..config import KNOWLEDGE_CANDIDATE_LIMIT, KNOWLEDGE_DB_PATH, KNOWLEDGE_RRF_K
from .embedder import OllamaEmbedder
from .evaluation import EvalCase, EvalReport, evaluate_retriever
from .retrieval import HybridRetriever, KeywordRetriever, VectorRetriever
from .store import VectorStore

_DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"


def _load_cases(path: Path) -> list[EvalCase]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [EvalCase(query=c["query"], expected_document_id=c["expected_document_id"]) for c in raw]


def _print_report(report: EvalReport) -> None:
    print(f"{report.name:<10} top1={report.top1:6.1f}%  top3={report.top3:6.1f}%  top5={report.top5:6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vector/keyword/hybrid retrieval accuracy.")
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Path to a JSON eval-cases file")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if not cases:
        print(f"No eval cases found in {args.cases}")
        return

    store = VectorStore(KNOWLEDGE_DB_PATH)
    embedder = OllamaEmbedder()
    vector_retriever = VectorRetriever(store, embedder)
    keyword_retriever = KeywordRetriever(store)
    hybrid_retriever = HybridRetriever(
        vector_retriever, keyword_retriever, k=KNOWLEDGE_RRF_K, candidate_limit=KNOWLEDGE_CANDIDATE_LIMIT
    )

    print(f"Evaluating {len(cases)} case(s) from {args.cases}")
    print()
    print(f"{'':<10} {'Top-1':>11} {'Top-3':>11} {'Top-5':>11}")
    _print_report(evaluate_retriever("vector", cases, vector_retriever.search))
    _print_report(evaluate_retriever("keyword", cases, keyword_retriever.search))
    _print_report(evaluate_retriever("hybrid", cases, hybrid_retriever.search))


if __name__ == "__main__":
    main()
