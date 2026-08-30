from .base import Tool, ToolResult, ToolValidationError
from .calculator import CalculatorTool
from .registry import ToolRegistry
from .search import SearchTool
from .time import TimeTool


def default_registry() -> ToolRegistry:
    """The standard set of tools Jarvis exposes to the LLM."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(SearchTool())
    return registry


__all__ = [
    "Tool",
    "ToolResult",
    "ToolValidationError",
    "ToolRegistry",
    "CalculatorTool",
    "TimeTool",
    "SearchTool",
    "default_registry",
]
