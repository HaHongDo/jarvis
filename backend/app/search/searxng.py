import logging
from typing import Any, Optional

import requests

from ..config import SEARCH_LANGUAGE, SEARCH_SAFESEARCH, SEARCH_TIMEOUT_SECONDS, SEARXNG_URL

logger = logging.getLogger(__name__)


class SearXNGError(Exception):
    """Raised when SearXNG cannot be reached or returns an unexpected response."""


class SearXNGTimeoutError(SearXNGError):
    """Raised when the SearXNG request times out."""


class SearXNGClient:
    """Thin HTTP client for a local SearXNG instance's JSON search API.

    Responsible only for talking to SearXNG over HTTP - parsing/normalizing the
    response into domain models happens in `SearchNormalizer`.
    """

    def __init__(self, base_url: str = SEARXNG_URL, timeout: float = SEARCH_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(
        self,
        query: str,
        *,
        category: str = "general",
        language: str = SEARCH_LANGUAGE,
        time_range: Optional[str] = None,
        page: int = 1,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": category,
            "language": language,
            "pageno": page,
            "safesearch": SEARCH_SAFESEARCH,
        }
        if time_range:
            params["time_range"] = time_range

        logger.info("SEARXNG request query=%r", query)
        try:
            response = requests.get(f"{self.base_url}/search", params=params, timeout=self.timeout)
        except requests.Timeout as exc:
            raise SearXNGTimeoutError("SearXNG request timed out.") from exc
        except requests.RequestException as exc:
            raise SearXNGError(f"Unable to reach SearXNG: {exc}") from exc

        if response.status_code != 200:
            raise SearXNGError(f"SearXNG returned HTTP {response.status_code}.")

        try:
            return response.json()
        except ValueError as exc:
            raise SearXNGError("SearXNG returned a non-JSON response.") from exc
