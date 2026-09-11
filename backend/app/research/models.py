from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Source:
    """A single deduplicated, fetched web source: the original search-discovery
    metadata (title/snippet) plus the extracted webpage content. This is what a
    `WebPage` becomes once it's folded into the research pipeline (see Day 10 plan)."""

    id: str
    url: str
    title: str
    content: str
    snippet: str = ""
    source_type: str = "web"
    published_at: Optional[str] = None
    fetched_at: Optional[datetime] = None
    rank: int = 0


@dataclass
class SourceChunk:
    """A slice of a `Source`'s content, small enough to fit a context budget while
    still traceable back to the source it came from via `source_id`."""

    source_id: str
    chunk_id: str
    text: str
    position: int


@dataclass
class ResearchContext:
    """The final research artifact: the query, every source that survived
    fetching/dedup, the chunks selected for the context budget, and the rendered
    context string itself (see Day 10 plan)."""

    query: str
    sources: list[Source] = field(default_factory=list)
    chunks: list[SourceChunk] = field(default_factory=list)
    context_text: str = ""
    created_at: Optional[datetime] = None
