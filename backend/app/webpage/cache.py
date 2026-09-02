import hashlib
import threading
import time
from typing import Optional

from ..search.normalizer import normalize_url


def _build_key(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


class PageCache:
    """In-memory TTL cache keyed by normalized URL, storing fetched-and-extracted
    `WebPage` objects. Mirrors `SearchCache`'s shape so the two caches behave
    consistently, but keys on the URL rather than a query + parameters."""

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, url: str) -> Optional[object]:
        key = _build_key(url)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, url: str, value: object, ttl_seconds: Optional[float] = None) -> None:
        key = _build_key(url)
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        with self._lock:
            self._store[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
