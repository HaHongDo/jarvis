"""CLI entry point for querying the knowledge base directly (see Day 12 plan item 15):

    python -m app.knowledge_search "How does authz_version work?"
    python -m app.knowledge_search "How does authz_version work?" --debug
    python -m app.knowledge_search "authentication" --project jarvis --limit 3

Without `--debug`, prints the final hybrid (RRF) results. With `--debug`, also prints
the vector-only and keyword-only candidate lists, so it's possible to see why a
particular chunk was (or wasn't) retrieved.
"""

import argparse

from .config import KNOWLEDGE_CANDIDATE_LIMIT, KNOWLEDGE_DB_PATH, KNOWLEDGE_RRF_K, KNOWLEDGE_TOP_K
from .knowledge import (
    HybridRetriever,
    KeywordRetriever,
    OllamaEmbedder,
    SearchResult,
    VectorRetriever,
    VectorStore,
)


def _print_results(heading: str, results: list[SearchResult], score_label: str) -> None:
    print(heading)
    print("-" * len(heading))
    print()
    if not results:
        print("(no results)")
        print()
        return
    for result in results:
        print(f"{result.rank}. {result.path}")
        print(f"   {score_label}: {result.score:.4f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Jarvis's local knowledge base.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--debug", action="store_true", help="Also show vector/keyword candidate lists")
    parser.add_argument("--limit", type=int, default=KNOWLEDGE_TOP_K, help="Number of final results to return")
    parser.add_argument("--project", help="Filter to documents with this metadata project")
    parser.add_argument("--source-type", help="Filter to documents with this metadata source_type")
    parser.add_argument("--document-type", help="Filter to documents with this metadata document_type")
    parser.add_argument("--owner-id", help="Filter to documents with this metadata owner_id")
    args = parser.parse_args()

    filters = {
        key: value
        for key, value in {
            "project": args.project,
            "source_type": args.source_type,
            "document_type": args.document_type,
            "owner_id": args.owner_id,
        }.items()
        if value is not None
    } or None

    store = VectorStore(KNOWLEDGE_DB_PATH)
    embedder = OllamaEmbedder()
    vector_retriever = VectorRetriever(store, embedder)
    keyword_retriever = KeywordRetriever(store)
    hybrid_retriever = HybridRetriever(
        vector_retriever, keyword_retriever, k=KNOWLEDGE_RRF_K, candidate_limit=KNOWLEDGE_CANDIDATE_LIMIT
    )

    print("QUERY")
    print("-----")
    print()
    print(args.query)
    print()
    print()

    if args.debug:
        vector_results = vector_retriever.search(args.query, limit=KNOWLEDGE_CANDIDATE_LIMIT, filters=filters)
        keyword_results = keyword_retriever.search(args.query, limit=KNOWLEDGE_CANDIDATE_LIMIT, filters=filters)
        _print_results("VECTOR RESULTS", vector_results, "score")
        _print_results("KEYWORD RESULTS", keyword_results, "score")

    final_results = hybrid_retriever.search(args.query, limit=args.limit, filters=filters)
    _print_results("RRF RESULTS" if args.debug else "RESULTS", final_results, "rrf" if args.debug else "score")


if __name__ == "__main__":
    main()
