import logging
from urllib.parse import urljoin

import requests

from ..config import PAGE_FETCH_TIMEOUT_SECONDS, PAGE_MAX_REDIRECTS, PAGE_MAX_RESPONSE_BYTES
from .ssrf import assert_safe_url

logger = logging.getLogger(__name__)

_USER_AGENT = "Jarvis/0.1 (+local-assistant; single-page-fetcher)"
_ALLOWED_CONTENT_TYPES = ("text/html",)
_REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)
_CHUNK_SIZE = 8192


class PageFetchError(Exception):
    """Raised when a page cannot be fetched at all (network error, timeout, bad status)."""


class PageTooLargeError(PageFetchError):
    """Raised when the response exceeds the maximum allowed size."""


class UnsupportedContentTypeError(PageFetchError):
    """Raised when the response content type isn't supported for extraction."""


class PageFetcher:
    """Fetches a single URL over HTTP(S) with SSRF protection, a redirect limit, a
    response-size cap, and a content-type allowlist.

    Redirects are followed manually (rather than relying on `requests`' built-in
    redirect handling) so every hop is re-validated by `assert_safe_url` - a redirect
    to an internal address must be rejected just like a direct request to one.
    """

    def __init__(
        self,
        timeout: float = PAGE_FETCH_TIMEOUT_SECONDS,
        max_redirects: int = PAGE_MAX_REDIRECTS,
        max_response_bytes: int = PAGE_MAX_RESPONSE_BYTES,
    ):
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_response_bytes = max_response_bytes

    def fetch(self, url: str) -> str:
        """Returns the response body as text. Raises a `PageFetchError` subclass on failure."""
        current_url = url

        for hop in range(self.max_redirects + 1):
            assert_safe_url(current_url)

            response = self._request(current_url)
            try:
                if response.status_code in _REDIRECT_STATUS_CODES:
                    current_url = self._resolve_redirect(current_url, response)
                    continue

                if response.status_code != 200:
                    raise PageFetchError(f"{current_url} returned HTTP {response.status_code}.")

                self._check_content_type(current_url, response)
                self._check_declared_length(current_url, response)
                return self._read_body(current_url, response)
            finally:
                response.close()

        raise PageFetchError(f"Too many redirects while fetching {url} (max {self.max_redirects}).")

    def _request(self, url: str) -> requests.Response:
        try:
            return requests.get(
                url,
                timeout=self.timeout,
                allow_redirects=False,
                stream=True,
                headers={"User-Agent": _USER_AGENT},
            )
        except requests.Timeout as exc:
            raise PageFetchError(f"Request to {url} timed out.") from exc
        except requests.RequestException as exc:
            raise PageFetchError(f"Unable to fetch {url}: {exc}") from exc

    def _resolve_redirect(self, current_url: str, response: requests.Response) -> str:
        location = response.headers.get("Location")
        if not location:
            raise PageFetchError(f"Redirect from {current_url} had no Location header.")
        return urljoin(current_url, location)

    def _check_content_type(self, url: str, response: requests.Response) -> None:
        content_type = response.headers.get("Content-Type", "")
        if not any(content_type.startswith(allowed) for allowed in _ALLOWED_CONTENT_TYPES):
            raise UnsupportedContentTypeError(
                f"Unsupported content type for {url}: {content_type or 'unknown'}"
            )

    def _check_declared_length(self, url: str, response: requests.Response) -> None:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > self.max_response_bytes:
            raise PageTooLargeError(f"{url} response too large ({content_length} bytes).")

    def _read_body(self, url: str, response: requests.Response) -> str:
        body = bytearray()
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            body.extend(chunk)
            if len(body) > self.max_response_bytes:
                raise PageTooLargeError(f"{url} response exceeded max size while streaming.")

        encoding = response.encoding or "utf-8"
        return bytes(body).decode(encoding, errors="replace")
