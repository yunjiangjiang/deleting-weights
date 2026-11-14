"""Exact rational AST verifier for Countdown arithmetic search tasks."""

from __future__ import annotations

import ast
import json
import re
from fractions import Fraction


def _eval_ast(node: ast.AST, used_nums: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, used_nums)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        used_nums.append(int(node.value))
        return Fraction(int(node.value), 1)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, used_nums)
        right = _eval_ast(node.right, used_nums)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def compute_score(solution_str: str, ground_truth: str | dict) -> float:
    """Return 1.0 iff <answer>...</answer> uses the exact multiset of `nums` via +,-,*,/ to equal `target`."""
    meta = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL | re.IGNORECASE))
    if not matches:
        return 0.0
    expr = matches[-1].group(1).strip()
    if "=" in expr:
        expr = expr.split("=")[0].strip()
    if not expr or not re.match(r"^[\d+\-*/()\s]+$", expr):
        return 0.0
    try:
        tree = ast.parse(expr, mode="eval")
        used_nums: list[int] = []
        val = _eval_ast(tree, used_nums)
    except Exception:
        return 0.0
    if sorted(used_nums) != sorted(int(x) for x in meta["nums"]):
        return 0.0
    return 1.0 if val == Fraction(int(meta["target"]), 1) else 0.0
