from .base import Tool, ToolResult, ToolValidationError
from .calculator import CalculatorTool
from .fetch_page import FetchPageTool
from .knowledge import KnowledgeSearchTool
from .registry import ToolRegistry
from .research import ResearchTool
from .search import SearchTool
from .time import TimeTool


def default_registry() -> ToolRegistry:
    """The standard set of tools Jarvis exposes to the LLM."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(SearchTool())
    registry.register(FetchPageTool())
    registry.register(ResearchTool())
    registry.register(KnowledgeSearchTool())
    return registry


__all__ = [
    "Tool",
    "ToolResult",
    "ToolValidationError",
    "ToolRegistry",
    "CalculatorTool",
    "TimeTool",
    "SearchTool",
    "FetchPageTool",
    "ResearchTool",
    "KnowledgeSearchTool",
    "default_registry",
]
