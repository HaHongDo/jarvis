from typing import Any, Optional

from ..knowledge import KnowledgeService
from .base import Tool


class KnowledgeSearchTool(Tool):
    """Semantic search over Jarvis's private local knowledge base (see Day 11 plan).

    Unlike `search`/`research`, this never touches the internet - it's a separate,
    private retrieval path over the user's own notes/docs (see Day 11 plan item 21:
    protect private knowledge). The LLM only sees ranked, attributed chunks, never
    the raw vector store.
    """

    name = "search_knowledge"
    description = (
        "Search Jarvis's private local knowledge base (the user's own notes, docs, "
        "and project files) for relevant information.\n\n"
        "Use this tool for questions like 'what did I write about...', 'what does "
        "my documentation say about...', 'my notes on...', or 'our architecture/"
        "project'. Do not use it for current events, news, prices, or anything that "
        "requires up-to-date information from the internet - use `search` or "
        "`research` for that instead. If a question needs both (e.g. 'what did I "
        "write about X, and is that still the recommended approach?'), call this "
        "tool and a web tool separately and combine the results.\n\n"
        "Returned notes are the user's own content, but always cite them by title/"
        "path when answering (e.g. 'According to your notes on X...')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused query describing what to look for in the knowledge base.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, service: Optional[KnowledgeService] = None):
        self.service = service or KnowledgeService()

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments["query"]
        matches = self.service.search(query)

        if not matches:
            return "No relevant information was found in your private knowledge base."

        return self.service.build_context(query, matches)
