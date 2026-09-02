import json
import logging
from typing import Optional

import trafilatura

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when no readable content could be extracted from a page."""


class ContentExtractor:
    """Extracts the main readable content (title/text/published date) from raw HTML.

    Delegates the actual parsing to trafilatura, which strips navigation, headers,
    footers, ads, cookie banners, and script/style content, leaving only the
    article/document body.
    """

    def extract(self, html: str, url: str) -> dict:
        raw = trafilatura.extract(
            html,
            url=url,
            output_format="json",
            with_metadata=True,
            favor_precision=True,
        )
        if not raw:
            raise ExtractionError(f"Could not extract readable content from {url}.")

        data = json.loads(raw)
        text = (data.get("text") or "").strip()
        if not text:
            raise ExtractionError(f"Extracted content from {url} was empty.")

        return {
            "title": data.get("title") or url,
            "text": text,
            "published_at": self._first_non_empty(data.get("date")),
        }

    @staticmethod
    def _first_non_empty(value: Optional[str]) -> Optional[str]:
        return value or None
