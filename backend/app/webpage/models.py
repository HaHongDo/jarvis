from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class WebPage:
    """A single fetched-and-extracted webpage. Never expose raw HTML or extractor
    output directly to the LLM - this is the stable interface the rest of Jarvis uses."""

    url: str
    title: str
    text: str
    published_at: Optional[str] = None
    fetched_at: Optional[datetime] = None
    cached: bool = False
