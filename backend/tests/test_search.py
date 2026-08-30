import time

import pytest

from app.search import SearchCache, SearchContextBuilder, SearchNormalizer, SearchService, SearchResult
from app.search.normalizer import normalize_query, normalize_url
from app.search.searxng import SearXNGError, SearXNGTimeoutError


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  Latest   Rust Release  ") == "latest rust release"


def test_normalize_url_strips_tracking_params_and_trailing_slash():
    a = normalize_url("https://Example.com/Page/?utm_source=x&id=1")
    b = normalize_url("https://example.com/Page?id=1")

    assert a == b


def test_normalizer_deduplicates_by_normalized_url():
    raw = {
        "results": [
            {"title": "A", "url": "https://example.com/a?utm_source=x", "content": "first"},
            {"title": "A dup", "url": "https://example.com/a", "content": "second"},
            {"title": "B", "url": "https://example.com/b", "content": "third"},
        ]
    }

    results = SearchNormalizer().normalize(raw)

    assert [r.url for r in results] == [
        "https://example.com/a?utm_source=x",
        "https://example.com/b",
    ]


def test_normalizer_skips_malformed_and_missing_fields():
    raw = {
        "results": [
            {"title": "No URL", "content": "..."},
            {"url": "https://example.com/no-title", "content": "..."},
            "not a dict",
            {"title": "Fine", "url": "https://example.com/fine", "content": "ok"},
        ]
    }

    results = SearchNormalizer().normalize(raw)

    assert len(results) == 1
    assert results[0].title == "Fine"


def test_normalizer_handles_empty_results():
    assert SearchNormalizer().normalize({}) == []
    assert SearchNormalizer().normalize({"results": []}) == []


def test_normalizer_respects_limit_before_dedup():
    raw = {
        "results": [
            {"title": f"T{i}", "url": f"https://example.com/{i}", "content": "x"} for i in range(10)
        ]
    }

    results = SearchNormalizer().normalize(raw, limit=3)

    assert len(results) == 3


def test_context_builder_formats_results():
    results = [
        SearchResult(title="Rust", url="https://rust-lang.org", snippet="A language.", source="rust-lang.org")
    ]

    context = SearchContextBuilder().build("rust", results)

    assert "## Web Search Results" in context
    assert "Query: rust" in context
    assert "Title: Rust" in context
    assert "URL: https://rust-lang.org" in context


def test_context_builder_warns_content_is_untrusted():
    results = [SearchResult(title="Rust", url="https://rust-lang.org", snippet="A language.")]

    context = SearchContextBuilder().build("rust", results)

    assert "untrusted" in context.lower()


def test_context_builder_handles_no_results():
    context = SearchContextBuilder().build("nothing useful", [])

    assert "No results found" in context


def test_cache_set_and_get_roundtrip():
    cache = SearchCache(ttl_seconds=60)
    cache.set("Latest Rust Release", "value")

    assert cache.get("latest   rust release") == "value"


def test_cache_expires_after_ttl():
    cache = SearchCache(ttl_seconds=0.01)
    cache.set("query", "value")
    time.sleep(0.02)

    assert cache.get("query") is None


def test_cache_distinguishes_different_queries():
    cache = SearchCache(ttl_seconds=60)
    cache.set("current bitcoin price", "now")
    cache.set("bitcoin price in 2020", "then")

    assert cache.get("current bitcoin price") == "now"
    assert cache.get("bitcoin price in 2020") == "then"


def test_cache_distinguishes_by_extra_params():
    cache = SearchCache(ttl_seconds=60)
    cache.set("go release", "today", time_range="day")
    cache.set("go release", "this year", time_range="year")

    assert cache.get("go release", time_range="day") == "today"
    assert cache.get("go release", time_range="year") == "this year"
    assert cache.get("go release") is None


def test_cache_set_ttl_override_expires_independently_of_default_ttl():
    cache = SearchCache(ttl_seconds=60)
    cache.set("query", "value", ttl_seconds=0.01)
    time.sleep(0.02)

    assert cache.get("query") is None


class _FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def search(self, query, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


def test_service_returns_normalized_results_and_caches():
    payload = {"results": [{"title": "Rust", "url": "https://rust-lang.org", "content": "..."}]}
    client = _FakeClient(payload=payload)
    service = SearchService(client=client, cache=SearchCache(ttl_seconds=60))

    response = service.search("rust")
    assert response.result_count == 1
    assert not response.cached

    cached_response = service.search("rust")
    assert cached_response.cached
    assert client.calls == 1


def test_service_propagates_searxng_errors():
    client = _FakeClient(error=SearXNGError("down"))
    service = SearchService(client=client, cache=SearchCache(ttl_seconds=60))

    with pytest.raises(SearXNGError):
        service.search("rust")


def test_service_propagates_timeout_errors():
    client = _FakeClient(error=SearXNGTimeoutError("timed out"))
    service = SearchService(client=client, cache=SearchCache(ttl_seconds=60))

    with pytest.raises(SearXNGTimeoutError):
        service.search("rust")


def test_service_cache_key_distinguishes_time_range():
    payload = {"results": [{"title": "Rust", "url": "https://rust-lang.org", "content": "..."}]}
    client = _FakeClient(payload=payload)
    service = SearchService(client=client, cache=SearchCache(ttl_seconds=60))

    service.search("rust release", time_range="day")
    response = service.search("rust release", time_range="year")

    assert not response.cached
    assert client.calls == 2


def test_service_realtime_freshness_bypasses_cache():
    payload = {"results": [{"title": "BTC", "url": "https://example.com", "content": "..."}]}
    client = _FakeClient(payload=payload)
    service = SearchService(client=client, cache=SearchCache(ttl_seconds=60))

    service.search("bitcoin price", freshness="realtime")
    response = service.search("bitcoin price", freshness="realtime")

    assert not response.cached
    assert client.calls == 2


def test_service_normal_freshness_still_caches():
    payload = {"results": [{"title": "Rust", "url": "https://rust-lang.org", "content": "..."}]}
    client = _FakeClient(payload=payload)
    service = SearchService(client=client, cache=SearchCache(ttl_seconds=60))

    service.search("rust", freshness="normal")
    response = service.search("rust", freshness="normal")

    assert response.cached
    assert client.calls == 1
