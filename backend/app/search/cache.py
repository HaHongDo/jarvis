import hashlib
import threading
import time
from typing import Optional

from .normalizer import normalize_query


def _build_key(query: str, **params: object) -> str:
    """Cache key = hash(normalized query + any extra search parameters).

    Including parameters like language/category/time_range prevents searches with
    different freshness requirements (e.g. "bitcoin price" vs "bitcoin price"
    with time_range=day) from incorrectly sharing cached results.
    """
    key_parts = [f"query={normalize_query(query)}"]
    for name in sorted(params):
        value = params[name]
        if value is not None:
            key_parts.append(f"{name}={value}")
    raw_key = "|".join(key_parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class SearchCache:
    """In-memory TTL cache keyed by a hash of the normalized query plus search
    parameters.

    Deliberately simple: exact (normalized) query + parameter matching only. Two
    semantically similar but factually different queries (e.g. "current
    Bitcoin price" vs "Bitcoin price in 2020") must not collide, so no
    semantic matching is attempted here - that's a later improvement.
    """

    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, query: str, **params: object) -> Optional[object]:
        key = _build_key(query, **params)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() >= expires_at:
                del self._store[key]
                return None
            return value

    def set(self, query: str, value: object, ttl_seconds: Optional[float] = None, **params: object) -> None:
        """Store `value` under the query+params key. `ttl_seconds` overrides the
        cache's default TTL, letting callers apply freshness-aware expiry per entry."""
        key = _build_key(query, **params)
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        with self._lock:
            self._store[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

