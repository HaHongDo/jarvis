import ast
import operator
from typing import Any

from .base import Tool

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluates a mathematical expression (e.g. '123 * 456') and returns the numeric result."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression using +, -, *, /, //, %, ** and parentheses.",
            }
        },
        "required": ["expression"],
    }

    def run(self, arguments: dict[str, Any]) -> float:
        expression = arguments["expression"]
        try:
            node = ast.parse(expression, mode="eval").body
            return _eval_node(node)
        except ToolExpressionError:
            raise
        except Exception as exc:
            raise ToolExpressionError(f"Invalid expression: {expression!r}") from exc


class ToolExpressionError(ValueError):
    """Raised when an expression cannot be safely evaluated."""


def _eval_node(node: ast.AST) -> float:
    """Evaluate a numeric-only AST node. Deliberately does not use eval() -
    a tool argument is untrusted model output and must never reach arbitrary code execution."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ToolExpressionError("Expression contains unsupported syntax.")
