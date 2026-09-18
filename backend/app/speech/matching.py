"""Shared phrase matching for the speech package.

Both the normalizer (deciding whether an alias occurs in a transcript) and the
vocabulary manager (deciding whether a keyword occurs in one) need the same
notion of "this phrase appears here": case-insensitive, on word boundaries, and
treating any run of whitespace/hyphens between tokens as interchangeable - so
"go routine", "go-routine" and "go  routine" share one pattern, while "routine
written in Go" matches nothing.
"""

import re
from functools import lru_cache

# Matches nothing, ever. Used for a degenerate (empty) alias so callers don't
# have to special-case it - an empty pattern body would otherwise compile to
# `\b\b`, which matches at every word boundary.
_NEVER = re.compile(r"(?!)")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def phrase_tokens(phrase: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", phrase.strip()) if t]


@lru_cache(maxsize=4096)
def phrase_pattern(phrase: str) -> re.Pattern:
    """Case-insensitive, word-boundary regex for `phrase`. Cached because the
    vocabulary manager re-matches the same few hundred keywords on every turn."""
    tokens = phrase_tokens(phrase)
    if not tokens:
        return _NEVER
    body = r"[\s-]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)
