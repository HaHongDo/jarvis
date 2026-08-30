import re
import threading
import time
from typing import Optional


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class SearchCache:
    """In-memory TTL cache keyed by normalized query text.

    Deliberately simple: exact (normalized) query matching only. Two
    semantically similar but factually different queries (e.g. "current
    Bitcoin price" vs "Bitcoin price in 2020") must not collide, so no
    semantic matching is attempted here - that's a later improvement.
    """

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, query: str) -> Optional[object]:
        key = _normalize_query(query)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, query: str, value: object) -> None:
        key = _normalize_query(query)
        with self._lock:
            self._store[key] = (time.time() + self.ttl_seconds, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
