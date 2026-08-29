from .base import Tool, ToolResult, ToolValidationError
from .calculator import CalculatorTool
from .fake_search import FakeSearchTool
from .registry import ToolRegistry
from .time import TimeTool


def default_registry() -> ToolRegistry:
    """The standard set of tools Jarvis exposes to the LLM."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(FakeSearchTool())
    return registry


__all__ = [
    "Tool",
    "ToolResult",
    "ToolValidationError",
    "ToolRegistry",
    "CalculatorTool",
    "TimeTool",
    "FakeSearchTool",
    "default_registry",
]
