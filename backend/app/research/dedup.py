import hashlib
import re

from ..search.normalizer import normalize_url
from .models import Source

__all__ = ["content_fingerprint", "dedupe_sources"]


def content_fingerprint(text: str) -> str:
    """A stable hash of normalized content, used to detect two different URLs that
    serve the same (or near-identical) article (see Day 10 plan: content dedup)."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedupe_sources(sources: list[Source]) -> list[Source]:
    """Removes sources with a duplicate normalized URL or a duplicate content
    fingerprint, keeping the first (highest-priority) occurrence of each."""
    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    deduped: list[Source] = []

    for source in sources:
        url_key = normalize_url(source.url)
        if url_key in seen_urls:
            continue

        fingerprint = content_fingerprint(source.content)
        if fingerprint in seen_fingerprints:
            continue

        seen_urls.add(url_key)
        seen_fingerprints.add(fingerprint)
        deduped.append(source)

    return deduped
