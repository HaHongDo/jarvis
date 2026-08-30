from typing import Any, Optional

from ..search import SearchService, SearXNGError, SearXNGTimeoutError
from .base import Tool


class SearchTool(Tool):
    """Real web-search tool backed by a local SearXNG instance.

    Replaces the Day 6 `FakeSearchTool` stub. Delegates all HTTP, caching,
    normalization, and formatting work to `SearchService`.
    """

    name = "search"
    description = (
        "Search the internet for current or externally verifiable information.\n\n"
        "Use this tool when:\n"
        "- the user asks for current information\n"
        "- the user asks about recent events\n"
        "- the answer is likely outside your knowledge\n"
        "- the user explicitly asks you to search the web\n\n"
        "Do not use this tool for:\n"
        "- basic reasoning\n"
        "- simple calculations\n"
        "- casual conversation\n"
        "- questions you can confidently answer without external information"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
        "required": ["query"],
    }

    def __init__(self, service: Optional[SearchService] = None):
        self.service = service or SearchService()

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments["query"]

        try:
            response = self.service.search(query)
        except SearXNGTimeoutError:
            return "I couldn't reach the web search service in time (it timed out)."
        except SearXNGError:
            return "I couldn't reach the web search service right now."

        if not response.results:
            return "I couldn't find any useful results for that."

        return self.service.build_context(response)
