import ipaddress
import socket
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = {"http", "https"}


class URLSafetyError(Exception):
    """Raised when a URL fails SSRF safety checks (bad scheme, unresolvable host, or
    resolves to a private/internal/reserved address)."""


def assert_safe_url(url: str) -> None:
    """Raise `URLSafetyError` unless `url` is a public http(s) URL.

    The LLM controls the URL passed to `fetch_page(url)`, so it must never be trusted.
    This resolves the hostname and rejects loopback, private, link-local, multicast, and
    other reserved address ranges - the same check is re-run for every redirect hop by
    `PageFetcher`, since a redirect to an internal address is just as dangerous as a
    direct request to one.
    """
    parts = urlsplit(url)

    if parts.scheme not in _ALLOWED_SCHEMES:
        raise URLSafetyError(f"Unsupported URL scheme: {parts.scheme!r}")

    hostname = parts.hostname
    if not hostname:
        raise URLSafetyError("URL has no hostname.")

    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise URLSafetyError(f"Could not resolve host: {hostname}") from exc

    if not addrinfo:
        raise URLSafetyError(f"Could not resolve host: {hostname}")

    for _family, _type, _proto, _canonname, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise URLSafetyError(f"Refusing to fetch unsafe/internal address: {ip}")
