from datetime import datetime, timezone

import pytest

from app.webpage import fetcher as fetcher_module
from app.webpage.cache import PageCache
from app.webpage.context import PageContextBuilder
from app.webpage.extractor import ContentExtractor, ExtractionError
from app.webpage.fetcher import PageFetchError, PageFetcher, PageTooLargeError, UnsupportedContentTypeError
from app.webpage.models import WebPage
from app.webpage.service import PageFetchService
from app.webpage.ssrf import URLSafetyError, assert_safe_url

# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


def test_assert_safe_url_rejects_non_http_scheme():
    with pytest.raises(URLSafetyError):
        assert_safe_url("ftp://example.com/file")


def test_assert_safe_url_rejects_missing_hostname():
    with pytest.raises(URLSafetyError):
        assert_safe_url("https:///no-host")


def test_assert_safe_url_rejects_unresolvable_host(monkeypatch):
    import socket

    import app.webpage.ssrf as ssrf_module

    def fake_getaddrinfo(host, port):
        raise socket.gaierror("not found")

    monkeypatch.setattr(ssrf_module.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(URLSafetyError):
        assert_safe_url("https://this-does-not-resolve.invalid")


def test_assert_safe_url_rejects_private_ip(monkeypatch):
    import app.webpage.ssrf as ssrf_module

    monkeypatch.setattr(
        ssrf_module.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("10.0.0.5", 0))]
    )

    with pytest.raises(URLSafetyError):
        assert_safe_url("https://internal.example.com")


def test_assert_safe_url_rejects_loopback(monkeypatch):
    import app.webpage.ssrf as ssrf_module

    monkeypatch.setattr(
        ssrf_module.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))]
    )

    with pytest.raises(URLSafetyError):
        assert_safe_url("https://localhost")


def test_assert_safe_url_allows_public_ip(monkeypatch):
    import app.webpage.ssrf as ssrf_module

    monkeypatch.setattr(
        ssrf_module.socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    assert_safe_url("https://example.com")


# ---------------------------------------------------------------------------
# PageFetcher
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, chunks=None, encoding="utf-8"):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []
        self.encoding = encoding
        self.closed = False

    def iter_content(self, chunk_size=8192):
        yield from self._chunks

    def close(self):
        self.closed = True


def test_fetcher_returns_body_text(monkeypatch):
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)
    response = _FakeResponse(headers={"Content-Type": "text/html; charset=utf-8"}, chunks=[b"<html>Hi</html>"])
    monkeypatch.setattr(fetcher_module.requests, "get", lambda *a, **k: response)

    text = PageFetcher().fetch("https://example.com")

    assert "Hi" in text


def test_fetcher_rejects_unsupported_content_type(monkeypatch):
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)
    response = _FakeResponse(headers={"Content-Type": "application/pdf"})
    monkeypatch.setattr(fetcher_module.requests, "get", lambda *a, **k: response)

    with pytest.raises(UnsupportedContentTypeError):
        PageFetcher().fetch("https://example.com/file.pdf")


def test_fetcher_rejects_declared_length_too_large(monkeypatch):
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)
    response = _FakeResponse(headers={"Content-Type": "text/html", "Content-Length": "999999999"})
    monkeypatch.setattr(fetcher_module.requests, "get", lambda *a, **k: response)

    with pytest.raises(PageTooLargeError):
        PageFetcher(max_response_bytes=1000).fetch("https://example.com")


def test_fetcher_rejects_streamed_body_too_large(monkeypatch):
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)
    response = _FakeResponse(headers={"Content-Type": "text/html"}, chunks=[b"a" * 2000])
    monkeypatch.setattr(fetcher_module.requests, "get", lambda *a, **k: response)

    with pytest.raises(PageTooLargeError):
        PageFetcher(max_response_bytes=1000).fetch("https://example.com")


def test_fetcher_raises_on_non_200_status(monkeypatch):
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)
    response = _FakeResponse(status_code=404, headers={"Content-Type": "text/html"})
    monkeypatch.setattr(fetcher_module.requests, "get", lambda *a, **k: response)

    with pytest.raises(PageFetchError):
        PageFetcher().fetch("https://example.com/missing")


def test_fetcher_follows_redirect_and_revalidates_each_hop(monkeypatch):
    checked_urls = []
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: checked_urls.append(url))

    responses = [
        _FakeResponse(status_code=302, headers={"Location": "https://example.com/final"}),
        _FakeResponse(headers={"Content-Type": "text/html"}, chunks=[b"final content"]),
    ]
    monkeypatch.setattr(fetcher_module.requests, "get", lambda *a, **k: responses.pop(0))

    text = PageFetcher().fetch("https://example.com/start")

    assert "final content" in text
    assert checked_urls == ["https://example.com/start", "https://example.com/final"]


def test_fetcher_raises_after_max_redirects(monkeypatch):
    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)
    monkeypatch.setattr(
        fetcher_module.requests,
        "get",
        lambda *a, **k: _FakeResponse(status_code=302, headers={"Location": "https://example.com/next"}),
    )

    with pytest.raises(PageFetchError):
        PageFetcher(max_redirects=2).fetch("https://example.com/start")


def test_fetcher_wraps_timeout(monkeypatch):
    import requests as requests_lib

    monkeypatch.setattr(fetcher_module, "assert_safe_url", lambda url: None)

    def raise_timeout(*args, **kwargs):
        raise requests_lib.Timeout("timed out")

    monkeypatch.setattr(fetcher_module.requests, "get", raise_timeout)

    with pytest.raises(PageFetchError):
        PageFetcher().fetch("https://example.com")


# ---------------------------------------------------------------------------
# ContentExtractor
# ---------------------------------------------------------------------------

_SAMPLE_HTML = """
<html>
<head><title>Test Article</title></head>
<body>
<nav>Home | About | Contact</nav>
<article>
<h1>Test Article</h1>
<p>This is the first paragraph of a fairly long article about testing content
extraction pipelines and how they behave with realistic body text that is long
enough to pass heuristics.</p>
<p>This is a second paragraph with more content to make sure trafilatura considers
this page a real article and not boilerplate navigation, adding even more filler
text here.</p>
</article>
<footer>Copyright 2026</footer>
</body>
</html>
"""


def test_extractor_extracts_main_text_and_strips_boilerplate():
    result = ContentExtractor().extract(_SAMPLE_HTML, "https://example.com/article")

    assert "first paragraph" in result["text"]
    assert "Home | About | Contact" not in result["text"]
    assert result["title"] == "Test Article"


def test_extractor_raises_on_empty_page():
    with pytest.raises(ExtractionError):
        ContentExtractor().extract("<html><body></body></html>", "https://example.com/empty")


# ---------------------------------------------------------------------------
# PageCache
# ---------------------------------------------------------------------------


def test_page_cache_set_and_get_roundtrip():
    cache = PageCache(ttl_seconds=60)
    cache.set("https://example.com/a", "value")

    assert cache.get("https://example.com/a") == "value"


def test_page_cache_normalizes_url_for_key():
    cache = PageCache(ttl_seconds=60)
    cache.set("https://Example.com/a/?utm_source=x", "value")

    assert cache.get("https://example.com/a") == "value"


def test_page_cache_expires_after_ttl():
    import time

    cache = PageCache(ttl_seconds=0.01)
    cache.set("https://example.com/a", "value")
    time.sleep(0.02)

    assert cache.get("https://example.com/a") is None


def test_page_cache_ttl_override_expires_independently():
    import time

    cache = PageCache(ttl_seconds=60)
    cache.set("https://example.com/a", "value", ttl_seconds=0.01)
    time.sleep(0.02)

    assert cache.get("https://example.com/a") is None


# ---------------------------------------------------------------------------
# PageContextBuilder
# ---------------------------------------------------------------------------


def test_context_builder_formats_page():
    page = WebPage(
        url="https://example.com",
        title="Example",
        text="Some content.",
        published_at="2026-01-01",
        fetched_at=datetime.now(timezone.utc),
    )

    context = PageContextBuilder().build(page, source_id=2)

    assert "SOURCE 2:" in context
    assert "Title: Example" in context
    assert "URL: https://example.com" in context
    assert "Published: 2026-01-01" in context
    assert "Some content." in context
    assert "untrusted" in context.lower()


# ---------------------------------------------------------------------------
# PageFetchService
# ---------------------------------------------------------------------------


class _FakeFetcher:
    def __init__(self, html="<html>fake</html>"):
        self.html = html
        self.calls = 0

    def fetch(self, url):
        self.calls += 1
        return self.html


class _FakeExtractor:
    def __init__(self, result=None):
        self.result = result or {"title": "T", "text": "x" * 100, "published_at": None}

    def extract(self, html, url):
        return self.result


def test_service_fetches_extracts_and_caches():
    fetcher = _FakeFetcher()
    service = PageFetchService(fetcher=fetcher, extractor=_FakeExtractor(), cache=PageCache(ttl_seconds=60))

    page = service.fetch("https://example.com/a")
    assert page.title == "T"
    assert not page.cached

    cached_page = service.fetch("https://example.com/a")
    assert cached_page.cached
    assert fetcher.calls == 1


def test_service_truncates_content_over_max_chars():
    extractor = _FakeExtractor({"title": "T", "text": "a" * 100, "published_at": None})
    service = PageFetchService(
        fetcher=_FakeFetcher(), extractor=extractor, cache=PageCache(ttl_seconds=60), max_content_chars=10
    )

    page = service.fetch("https://example.com/a")

    assert len(page.text) == 10


def test_service_propagates_fetch_errors():
    class _FailingFetcher:
        def fetch(self, url):
            raise PageFetchError("boom")

    service = PageFetchService(
        fetcher=_FailingFetcher(), extractor=_FakeExtractor(), cache=PageCache(ttl_seconds=60)
    )

    with pytest.raises(PageFetchError):
        service.fetch("https://example.com/a")


def test_service_build_context_delegates_to_context_builder():
    page = WebPage(url="https://example.com", title="T", text="body", fetched_at=datetime.now(timezone.utc))
    service = PageFetchService(cache=PageCache(ttl_seconds=60))

    context = service.build_context(page, source_id=1)

    assert "SOURCE 1:" in context
    assert "Title: T" in context
