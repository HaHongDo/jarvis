import json
import logging
import time

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

_MAX_LOGGED_ARGUMENT_CHARS = 200


class ToolRegistry:
    """The controlled gateway through which the model accesses capabilities.

    The LLM never calls a tool directly - it only sees schemas from `schemas()` and
    every invocation is routed through `execute()`, which validates arguments and logs
    the call regardless of whether the tool it names exists.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolResult:
        start = time.perf_counter()
        tool = self.get(name)

        if tool is None:
            result = ToolResult(success=False, error=f"Unknown tool: {name}")
        else:
            result = tool.execute(arguments)

        duration_ms = (time.perf_counter() - start) * 1000
        self._log(name, arguments, duration_ms, result)
        return result

    def _log(self, name: str, arguments: dict, duration_ms: float, result: ToolResult) -> None:
        args_repr = json.dumps(arguments, default=str)
        if len(args_repr) > _MAX_LOGGED_ARGUMENT_CHARS:
            args_repr = args_repr[:_MAX_LOGGED_ARGUMENT_CHARS] + "...(truncated)"

        result_size = len(result.to_content())
        logger.info(
            "TOOL %s arguments=%s duration=%.0fms success=%s result_size=%d",
            name,
            args_repr,
            duration_ms,
            result.success,
            result_size,
        )
        if not result.success:
            logger.warning("TOOL %s failed: %s", name, result.error)
