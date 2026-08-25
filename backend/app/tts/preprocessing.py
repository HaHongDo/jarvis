import re

_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(.+?)\1")
_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_for_speech(text: str) -> str:
    """Strip Markdown formatting that should not be spoken literally."""
    text = _LINK_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    text = _BOLD_ITALIC_RE.sub(r"\2", text)
    text = _HEADER_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
