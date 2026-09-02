import copy
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ..config import PAGE_CACHE_TTL_SECONDS, PAGE_MAX_CONTENT_CHARS
from .cache import PageCache
from .context import PageContextBuilder
from .extractor import ContentExtractor, ExtractionError
from .fetcher import PageFetchError, PageFetcher, PageTooLargeError, UnsupportedContentTypeError
from .models import WebPage
from .ssrf import URLSafetyError

logger = logging.getLogger(__name__)

__all__ = [
    "PageFetchService",
    "PageFetchError",
    "PageTooLargeError",
    "UnsupportedContentTypeError",
    "URLSafetyError",
    "ExtractionError",
]


class PageFetchService:
    """Orchestrates the page-fetch pipeline: cache -> HTTP fetch -> extract -> truncate.

    Mirrors `SearchService`'s cache-first structure so the "discovery" (search) and
    "reading" (fetch_page) pipelines follow the same shape. `assert_safe_url`/redirect
    validation happens inside `PageFetcher`, so SSRF errors surface here unchanged.
    """

    def __init__(
        self,
        fetcher: Optional[PageFetcher] = None,
        extractor: Optional[ContentExtractor] = None,
        context_builder: Optional[PageContextBuilder] = None,
        cache: Optional[PageCache] = None,
        max_content_chars: int = PAGE_MAX_CONTENT_CHARS,
    ):
        self.fetcher = fetcher or PageFetcher()
        self.extractor = extractor or ContentExtractor()
        self.context_builder = context_builder or PageContextBuilder()
        self.cache = cache or PageCache(ttl_seconds=PAGE_CACHE_TTL_SECONDS)
        self.max_content_chars = max_content_chars

    def fetch(self, url: str) -> WebPage:
        cached = self.cache.get(url)
        if cached is not None:
            logger.info("PAGE cache hit url=%r", url)
            page = copy.copy(cached)
            page.cached = True
            return page

        start = time.perf_counter()
        html = self.fetcher.fetch(url)
        elapsed = time.perf_counter() - start
        logger.info("PAGE fetch latency=%.2fs url=%r size=%d", elapsed, url, len(html))

        extracted = self.extractor.extract(html, url)
        text = extracted["text"]
        truncated = len(text) > self.max_content_chars
        if truncated:
            text = text[: self.max_content_chars]

        page = WebPage(
            url=url,
            title=extracted["title"],
            text=text,
            published_at=extracted.get("published_at"),
            fetched_at=datetime.now(timezone.utc),
        )
        logger.info(
            "PAGE extracted url=%r title=%r text_len=%d truncated=%s",
            url,
            page.title,
            len(text),
            truncated,
        )

        self.cache.set(url, page, ttl_seconds=PAGE_CACHE_TTL_SECONDS)
        return page

    def build_context(self, page: WebPage, source_id: int = 1) -> str:
        return self.context_builder.build(page, source_id=source_id)
