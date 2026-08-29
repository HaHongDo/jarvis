import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

_JSON_TYPES: dict[str, tuple] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


class ToolValidationError(Exception):
    """Raised when model-supplied arguments don't match a tool's declared schema."""


@dataclass
class ToolResult:
    """Outcome of executing a tool, structured so the LLM can be told what happened either way."""

    success: bool
    result: Any = None
    error: Optional[str] = None

    def to_content(self) -> str:
        """Serialize for a {"role": "tool"} message sent back to the LLM."""
        if self.success:
            return json.dumps({"success": True, "result": self.result})
        return json.dumps({"success": False, "error": self.error})


class Tool(ABC):
    """A single capability the LLM may invoke, gated by name/description/parameters + validation.

    The LLM only ever sees `to_schema()`. It never gets to run arbitrary code -
    `execute()` validates arguments against `parameters` before `run()` is called.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool and return a JSON-serializable result, or raise on failure."""
        raise NotImplementedError

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolValidationError("Arguments must be an object.")

        properties = self.parameters.get("properties", {})
        for required_name in self.parameters.get("required", []):
            if required_name not in arguments:
                raise ToolValidationError(f"Missing required argument: {required_name}")

        for arg_name, arg_value in arguments.items():
            schema = properties.get(arg_name)
            if schema is None:
                continue
            expected_type = schema.get("type")
            allowed_types = _JSON_TYPES.get(expected_type)
            if allowed_types and not isinstance(arg_value, allowed_types):
                raise ToolValidationError(
                    f"Argument '{arg_name}' must be of type '{expected_type}', "
                    f"got '{type(arg_value).__name__}'."
                )

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            self.validate_arguments(arguments)
            result = self.run(arguments)
        except ToolValidationError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
        return ToolResult(success=True, result=result)

    def to_schema(self) -> dict[str, Any]:
        """The provider-agnostic function-calling schema the LLM sees for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
