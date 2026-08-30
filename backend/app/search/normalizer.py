import re
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import SearchResult

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
    "mc_cid",
    "mc_eid",
}


def normalize_query(query: str) -> str:
    """Collapse whitespace and lowercase a query so equivalent queries (e.g. differing
    only in case or spacing) compare equal for cache keys and duplicate-search detection."""
    return re.sub(r"\s+", " ", query.strip().lower())


def normalize_url(url: str) -> str:
    """Strip tracking params, lowercase the host, and drop trailing slashes so
    equivalent URLs compare equal for deduplication."""
    parts = urlsplit(url)
    query_pairs = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING_PARAMS]
    path = parts.path.rstrip("/") or "/"
    netloc = parts.netloc.lower()
    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(query_pairs), ""))


class SearchNormalizer:
    """Turns raw SearXNG JSON into a clean, deduplicated list of `SearchResult`.

    Handles missing titles/URLs/snippets, duplicate URLs, empty results, and
    malformed entries so nothing raw ever reaches the LLM.
    """

    def normalize(self, response: dict[str, Any], limit: Optional[int] = None) -> list[SearchResult]:
        raw_results = response.get("results") or []
        if limit is not None:
            raw_results = raw_results[:limit]

        seen_urls: set[str] = set()
        normalized: list[SearchResult] = []

        for raw in raw_results:
            if not isinstance(raw, dict):
                continue

            url = raw.get("url")
            title = raw.get("title")
            if not url or not title:
                continue

            deduped_url = normalize_url(url)
            if deduped_url in seen_urls:
                continue
            seen_urls.add(deduped_url)

            normalized.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=raw.get("content") or "",
                    source=urlsplit(url).netloc or None,
                    published_at=raw.get("publishedDate"),
                )
            )

        return normalized
