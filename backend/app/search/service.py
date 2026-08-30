import copy
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ..config import SEARCH_CACHE_TTL_SECONDS, SEARCH_FETCH_RESULTS, SEARCH_MAX_RESULTS
from .cache import SearchCache
from .context import SearchContextBuilder
from .models import SearchResponse
from .normalizer import SearchNormalizer
from .searxng import SearXNGClient, SearXNGError, SearXNGTimeoutError

logger = logging.getLogger(__name__)

__all__ = ["SearchService", "SearXNGError", "SearXNGTimeoutError"]


class SearchService:
    """Orchestrates the search pipeline: cache -> SearXNG -> normalize -> limit -> context.

    This is the single entry point the search tool calls; it owns the cache and
    wires the client, normalizer, and context builder together.
    """

    def __init__(
        self,
        client: Optional[SearXNGClient] = None,
        normalizer: Optional[SearchNormalizer] = None,
        context_builder: Optional[SearchContextBuilder] = None,
        cache: Optional[SearchCache] = None,
        max_results: int = SEARCH_MAX_RESULTS,
        fetch_results: int = SEARCH_FETCH_RESULTS,
    ):
        self.client = client or SearXNGClient()
        self.normalizer = normalizer or SearchNormalizer()
        self.context_builder = context_builder or SearchContextBuilder()
        self.cache = cache or SearchCache(ttl_seconds=SEARCH_CACHE_TTL_SECONDS)
        self.max_results = max_results
        self.fetch_results = fetch_results

    def search(self, query: str) -> SearchResponse:
        cached = self.cache.get(query)
        if cached is not None:
            logger.info("SEARCH cache hit query=%r", query)
            response = copy.copy(cached)
            response.cached = True
            return response

        start = time.perf_counter()
        raw = self.client.search(query)
        elapsed = time.perf_counter() - start
        logger.info("SEARCH SearXNG latency=%.2fs query=%r", elapsed, query)

        normalized = self.normalizer.normalize(raw, limit=self.fetch_results)
        results = normalized[: self.max_results]

        response = SearchResponse(
            query=query,
            results=results,
            result_count=len(results),
            searched_at=datetime.now(timezone.utc).isoformat(),
        )
        self.cache.set(query, response)
        return response

    def build_context(self, response: SearchResponse) -> str:
        return self.context_builder.build(response.query, response.results)
