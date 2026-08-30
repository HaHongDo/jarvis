from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """A single, cleaned-up search result. Never expose raw SearXNG JSON to the LLM."""

    title: str
    url: str
    snippet: str
    source: Optional[str] = None
    published_at: Optional[str] = None


@dataclass
class SearchResponse:
    """Metadata-bearing result of a search, suitable for caching and logging."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    result_count: int = 0
    searched_at: str = ""
    cached: bool = False
