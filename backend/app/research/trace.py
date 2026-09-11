import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["ResearchTrace"]


@dataclass
class ResearchTrace:
    """Records each stage of a `ResearchService.research()` call for debugging
    (see Day 10 plan: research trace/logging)."""

    query: str
    cache_hit: bool = False
    result_count: int = 0
    selected_count: int = 0
    fetch_attempted: int = 0
    fetch_succeeded: int = 0
    fetch_failed: int = 0
    extracted_count: int = 0
    deduped_count: int = 0
    chunk_count: int = 0
    context_chunk_count: int = 0
    elapsed_seconds: float = 0.0
    timed_out: bool = False
    errors: list[str] = field(default_factory=list)

    def log(self) -> None:
        logger.info(
            "RESEARCH TRACE query=%r cache=%s results=%d selected=%d fetched=%d/%d "
            "extracted=%d deduped=%d chunks=%d context_chunks=%d elapsed=%.2fs "
            "timed_out=%s errors=%d",
            self.query,
            "HIT" if self.cache_hit else "MISS",
            self.result_count,
            self.selected_count,
            self.fetch_succeeded,
            self.fetch_attempted,
            self.extracted_count,
            self.deduped_count,
            self.chunk_count,
            self.context_chunk_count,
            self.elapsed_seconds,
            self.timed_out,
            len(self.errors),
        )
        for err in self.errors:
            logger.warning("RESEARCH TRACE error: %s", err)
