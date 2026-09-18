from .models import SearchResult

__all__ = ["KnowledgeContextBuilder"]


class KnowledgeContextBuilder:
    """Builds a budget-limited context string from ranked knowledge-base matches,
    each attributed to its source document (title/path) and similarity score, so
    the LLM can say "According to your notes on X..." (see Day 11 plan items 12,
    17, 20: provenance). Matches are already deduplicated by `chunk_id` before they
    reach here (Reciprocal Rank Fusion keys on it - see Day 12 plan item 20), and
    retrieval internals like `source`/`rank` are intentionally left out of the
    LLM-facing text (see Day 12 plan item 22)."""

    def __init__(self, max_context_chars: int):
        self.max_context_chars = max_context_chars

    def build(self, query: str, matches: list[SearchResult]) -> str:
        if not matches:
            return self._empty(query)

        lines = [
            "## Knowledge Base Results",
            "",
            f"Query: {query}",
            "",
        ]

        remaining = self.max_context_chars
        included = 0
        for match in matches:
            if len(match.content) > remaining:
                continue
            lines.append(f'<note title="{match.title}" path="{match.path}" score="{match.score:.2f}">')
            lines.append(match.content)
            lines.append("</note>")
            lines.append("")
            remaining -= len(match.content)
            included += 1

        if included == 0:
            return self._empty(query)

        return "\n".join(lines).rstrip() + "\n"

    def _empty(self, query: str) -> str:
        return (
            "## Knowledge Base Results\n\n"
            f"Query: {query}\n\n"
            "No relevant information was found in your private knowledge base.\n"
        )
