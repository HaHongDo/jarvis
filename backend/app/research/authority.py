from enum import IntEnum
from urllib.parse import urlsplit

__all__ = ["AuthorityTier", "classify_domain"]


class AuthorityTier(IntEnum):
    """Simple source-authority hierarchy (see Day 10 plan: source prioritization).
    Higher values should survive first when the context budget is limited."""

    LOW = 1
    MEDIUM = 2
    MEDIUM_HIGH = 3
    HIGH = 4


_HIGH_SUFFIXES = (".gov", ".edu")

# Official documentation / official project sites.
_HIGH_DOMAINS = {
    "docs.python.org",
    "go.dev",
    "golang.org",
    "developer.mozilla.org",
    "kubernetes.io",
    "redis.io",
    "postgresql.org",
    "www.postgresql.org",
    "docs.docker.com",
    "react.dev",
    "nodejs.org",
    "docs.github.com",
    "learn.microsoft.com",
    "docs.microsoft.com",
    "cloud.google.com",
    "docs.aws.amazon.com",
    "pytorch.org",
    "rust-lang.org",
    "www.rust-lang.org",
}

# Major/reputable news and technical publications.
_MEDIUM_HIGH_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "theguardian.com",
    "bloomberg.com",
    "wsj.com",
    "arstechnica.com",
    "techcrunch.com",
    "wired.com",
    "theverge.com",
}

# Forums / social media - useful context, but generally not authoritative.
_LOW_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "x.com",
    "twitter.com",
    "quora.com",
    "facebook.com",
    "news.ycombinator.com",
}


def classify_domain(url: str) -> AuthorityTier:
    """Heuristic authority classification by domain. Unknown domains default to
    MEDIUM rather than penalizing sources we simply don't recognize."""
    host = (urlsplit(url).netloc or "").lower().split(":")[0]
    bare_host = host[4:] if host.startswith("www.") else host

    if any(host.endswith(suffix) for suffix in _HIGH_SUFFIXES):
        return AuthorityTier.HIGH
    if host in _HIGH_DOMAINS or bare_host in _HIGH_DOMAINS:
        return AuthorityTier.HIGH
    if host.startswith("docs.") or host.startswith("developer."):
        return AuthorityTier.HIGH
    if host in _MEDIUM_HIGH_DOMAINS or bare_host in _MEDIUM_HIGH_DOMAINS:
        return AuthorityTier.MEDIUM_HIGH
    if host in _LOW_DOMAINS or bare_host in _LOW_DOMAINS:
        return AuthorityTier.LOW
    return AuthorityTier.MEDIUM
