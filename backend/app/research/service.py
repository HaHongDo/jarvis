import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Optional

from ..config import (
    RESEARCH_CHUNK_OVERLAP,
    RESEARCH_CHUNK_SIZE,
    RESEARCH_MAX_CONTEXT_CHARS,
    RESEARCH_MAX_SOURCES,
    RESEARCH_TIMEOUT_SECONDS,
)
from ..search import SearchResult, SearchService
from ..search.searxng import SearXNGError, SearXNGTimeoutError
from ..webpage import (
    ExtractionError,
    PageFetchError,
    PageFetchService,
    PageTooLargeError,
    UnsupportedContentTypeError,
    URLSafetyError,
)
from .authority import classify_domain
from .chunking import chunk_source
from .context import ResearchContextBuilder
from .dedup import dedupe_sources
from .models import ResearchContext, Source
from .trace import ResearchTrace

logger = logging.getLogger(__name__)

__all__ = ["ResearchService"]

_FETCH_ERRORS = (
    PageFetchError,
    PageTooLargeError,
    UnsupportedContentTypeError,
    ExtractionError,
    URLSafetyError,
)


class ResearchService:
    """Orchestrates the full source-aware research pipeline (see Day 10 plan):

        search -> select sources -> fetch pages -> extract -> dedupe -> chunk ->
        budget-limited context

    This runs entirely inside the application - `search()`, `fetch_page()`,
    chunking, deduplication, and the context budget are never exposed to the LLM
    individually; the `research` tool is the only LLM-facing surface for this
    pipeline (see Day 10 plan item 15).
    """

    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        page_service: Optional[PageFetchService] = None,
        context_builder: Optional[ResearchContextBuilder] = None,
        max_sources: int = RESEARCH_MAX_SOURCES,
        chunk_size: int = RESEARCH_CHUNK_SIZE,
        chunk_overlap: int = RESEARCH_CHUNK_OVERLAP,
        timeout_seconds: float = RESEARCH_TIMEOUT_SECONDS,
    ):
        self.search_service = search_service or SearchService()
        self.page_service = page_service or PageFetchService()
        self.context_builder = context_builder or ResearchContextBuilder(RESEARCH_MAX_CONTEXT_CHARS)
        self.max_sources = max_sources
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.timeout_seconds = timeout_seconds

    def research(
        self,
        query: str,
        *,
        time_range: Optional[str] = None,
        freshness: str = "normal",
    ) -> ResearchContext:
        start = time.perf_counter()
        trace = ResearchTrace(query=query)

        try:
            response = self.search_service.search(query, time_range=time_range, freshness=freshness)
        except (SearXNGError, SearXNGTimeoutError) as exc:
            trace.errors.append(f"search failed: {exc}")
            trace.elapsed_seconds = time.perf_counter() - start
            trace.log()
            return ResearchContext(query=query, created_at=datetime.now(timezone.utc))

        trace.cache_hit = response.cached
        trace.result_count = len(response.results)

        candidates = self._select_candidates(response.results)
        trace.selected_count = len(candidates)

        sources, timed_out = self._fetch_sources(candidates, trace, start)
        trace.timed_out = timed_out

        deduped = dedupe_sources(sources)
        trace.deduped_count = len(deduped)

        chunks = []
        for source in deduped:
            chunks.extend(chunk_source(source, chunk_size=self.chunk_size, overlap=self.chunk_overlap))
        trace.chunk_count = len(chunks)

        priority = {source.id: (-classify_domain(source.url).value, source.rank) for source in deduped}
        selected_chunks = self.context_builder.select_chunks(chunks, priority)
        trace.context_chunk_count = len(selected_chunks)

        context_text = self.context_builder.build(query, deduped, selected_chunks)

        trace.elapsed_seconds = time.perf_counter() - start
        trace.log()

        return ResearchContext(
            query=query,
            sources=deduped,
            chunks=selected_chunks,
            context_text=context_text,
            created_at=datetime.now(timezone.utc),
        )

    def _select_candidates(self, results: list[SearchResult]) -> list[SearchResult]:
        """Picks which search results to actually fetch: authority tier first (so a
        limited fetch budget favors official/authoritative domains), original search
        rank as the tiebreaker (see Day 10 plan: source prioritization)."""
        indexed = list(enumerate(results))
        indexed.sort(key=lambda pair: (-classify_domain(pair[1].url).value, pair[0]))
        return [result for _, result in indexed[: self.max_sources]]

    def _fetch_sources(
        self, candidates: list[SearchResult], trace: ResearchTrace, start: float
    ) -> tuple[list[Source], bool]:
        """Fetches every candidate concurrently, bounded by an overall research
        timeout. Fetch failures (network errors, SSRF rejection, unsupported content,
        extraction failure) are logged and skipped rather than failing the whole
        request (see Day 10 plan: partial failures, research timeout)."""
        trace.fetch_attempted = len(candidates)
        if not candidates:
            return [], False

        sources: list[Source] = []
        timed_out = False

        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            future_to_meta = {
                executor.submit(self.page_service.fetch, result.url): (rank, result)
                for rank, result in enumerate(candidates)
            }

            remaining = max(self.timeout_seconds - (time.perf_counter() - start), 0.1)
            done, not_done = wait(future_to_meta.keys(), timeout=remaining)

            if not_done:
                timed_out = True
                for future in not_done:
                    _rank, result = future_to_meta[future]
                    future.cancel()
                    trace.fetch_failed += 1
                    trace.errors.append(f"timed out fetching {result.url}")

            for future in done:
                rank, result = future_to_meta[future]
                try:
                    page = future.result()
                except _FETCH_ERRORS as exc:
                    trace.fetch_failed += 1
                    trace.errors.append(f"{result.url}: {exc}")
                    continue
                except Exception as exc:  # pragma: no cover - defensive
                    trace.fetch_failed += 1
                    trace.errors.append(f"{result.url}: unexpected error: {exc}")
                    continue

                trace.fetch_succeeded += 1
                trace.extracted_count += 1
                sources.append(
                    Source(
                        id=f"src_{rank + 1:03d}",
                        url=page.url,
                        title=page.title,
                        content=page.text,
                        snippet=result.snippet,
                        published_at=page.published_at,
                        fetched_at=page.fetched_at,
                        rank=rank,
                    )
                )

        sources.sort(key=lambda s: s.rank)
        return sources, timed_out
