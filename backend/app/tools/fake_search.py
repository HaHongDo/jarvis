from typing import Any

from .base import Tool


class FakeSearchTool(Tool):
    """Stub search tool. Proves the LLM -> tool -> result -> LLM loop before Day 7
    replaces this with a real SearXNG-backed search implementation."""

    name = "search"
    description = "Searches the web for information about a query and returns a list of results."
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

    def run(self, arguments: dict[str, Any]) -> list[dict[str, str]]:
        query = arguments["query"]
        return [
            {
                "title": f"Example result for '{query}'",
                "url": "https://example.com",
                "snippet": (
                    f"This is a placeholder search result about {query}. "
                    "Real web search is not implemented yet."
                ),
            }
        ]
