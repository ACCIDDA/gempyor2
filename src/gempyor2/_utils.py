# src/gempyor2/_utils.py
"""Utility functions for safe expression parsing and normalization in gempyor2."""

from __future__ import annotations

import ast
import operator as _op
from collections.abc import Iterable, Mapping
from typing import Any, Never

# =============================================================================
# Allowed AST node sets
# =============================================================================

_NUMERIC_ALLOWED = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
    ast.UAdd,
    ast.USub,
    ast.Tuple,
    ast.List,
    ast.Load,
}

_SYMBOLIC_ALLOWED = _NUMERIC_ALLOWED | {ast.Name, ast.Load}


class _SafeEvalError(ValueError):
    """Raised when an expression contains disallowed syntax or types."""


def _names_in_expr(expr: str) -> set[str]:
    """Return all variable names used in an expression.

    Args:
        expr: String expression to inspect.

    Returns:
        Unique variable names encountered in the expression.
    """
    if not isinstance(expr, str):
        return set()
    tree = ast.parse(expr, mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


# =============================================================================
# Minimal, auditable evaluator (no eval/exec)
# =============================================================================


class _Evaluator(ast.NodeVisitor):
    """Evaluate a vetted AST using only basic arithmetic.

    Notes:
        - No function calls, attributes, subscripts, comprehensions, lambdas.
        - Only numeric constants, names (when enabled), tuples/lists of numbers.
        - Operators: +, -, *, /, //, %, ** and unary +, -.
    """

    def __init__(  # FBT001: keyword-only boolean
        self,
        *,
        allow_names: bool,
        namespace: Mapping[str, float] | None = None,
    ) -> None:
        self.allow_names = allow_names
        self.ns = dict(namespace or {})

    def visit_Expression(self, node: ast.Expression) -> float | tuple | list:
        return self.visit(node.body)

    @staticmethod
    def visit_Constant(node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        msg = "Only numeric constants are allowed."
        raise _SafeEvalError(msg)

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(float(self.visit(elt)) for elt in node.elts)

    def visit_List(self, node: ast.List) -> list:
        return [float(self.visit(elt)) for elt in node.elts]

    def visit_Name(self, node: ast.Name) -> float:
        if not self.allow_names:
            msg = "Names are not allowed in this context."
            raise _SafeEvalError(msg)
        if node.id not in self.ns:
            msg = f"Unknown name: {node.id!r}"
            raise _SafeEvalError(msg)
        val = self.ns[node.id]
        if not isinstance(val, (int, float)):
            msg = f"Name {node.id!r} is not numeric."
            raise _SafeEvalError(msg)
        return float(val)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        val = float(self.visit(node.operand))
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        msg = "Only unary + and - are allowed."
        raise _SafeEvalError(msg)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = float(self.visit(node.left))
        right = float(self.visit(node.right))

        ops: dict[type, Any] = {
            ast.Add: _op.add,
            ast.Sub: _op.sub,
            ast.Mult: _op.mul,
            ast.Div: _op.truediv,
            ast.FloorDiv: _op.floordiv,
            ast.Mod: _op.mod,
            ast.Pow: _op.pow,
        }
        for op_type, fn in ops.items():
            if isinstance(node.op, op_type):
                return float(fn(left, right))

        msg = "Unsupported binary operator."
        raise _SafeEvalError(msg)

    # Deny anything else explicitly
    @staticmethod
    def generic_visit(node: ast.AST) -> Never:
        msg = f"Disallowed syntax: {type(node).__name__}"
        raise _SafeEvalError(msg)


def _validate_ast(tree: ast.AST, allowed: set[type]) -> None:
    for node in ast.walk(tree):
        if type(node) not in allowed:
            msg = f"Disallowed syntax: {type(node).__name__}"
            raise _SafeEvalError(msg)
        # Explicit guard in case allowed sets are extended later.
        if isinstance(
            node,
            (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.Set),
        ):
            msg = f"Disallowed syntax: {type(node).__name__}"
            raise _SafeEvalError(msg)


# =============================================================================
# Public safe-eval helpers
# =============================================================================


def _safe_eval_numeric(expr: str | float) -> float:
    """Safely evaluate a numeric expression; disallows names and function calls.

    Args:
        expr: A number or a numeric string expression (e.g., "1/5" or "2**3").

    Returns:
        The evaluated numeric value.

    Raises:
        TypeError: If `expr` is not a str|int|float.
        ValueError: If the expression contains disallowed syntax
            or cannot be evaluated.
    """
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str):
        msg = f"Expected str|int|float, got {type(expr)}"
        raise TypeError(msg)

    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree, _NUMERIC_ALLOWED)
    try:
        return float(_Evaluator(allow_names=False).visit(tree))
    except _SafeEvalError as e:
        msg = f"Invalid numeric expression: {expr!r} ({e})"
        raise ValueError(msg) from None


def _safe_eval_symbolic(expr: str, ns: Mapping[str, float]) -> float:
    """Safely evaluate arithmetic with variables present in `ns`.

    Allows only arithmetic ops and names present in `ns`.

    Args:
        expr: Symbolic arithmetic expression (e.g., "N - I0 - R0").
        ns: Mapping of allowed symbol names to numeric values.

    Returns:
        The evaluated result.

    Raises:
        TypeError: If inputs are of the wrong type.
        KeyError: If a name in the expression is not present in `ns`.
        ValueError: If the expression contains disallowed syntax
            or cannot be evaluated.
    """
    if not isinstance(expr, str):
        msg = "Expression must be a string."
        raise TypeError(msg)
    if not isinstance(ns, Mapping):
        msg = "Namespace must be a mapping of names to numbers."
        raise TypeError(msg)

    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree, _SYMBOLIC_ALLOWED)

    unknown = _names_in_expr(expr) - set(ns.keys())
    if unknown:
        msg = f"Unknown name(s) in expression: {sorted(unknown)}"
        raise KeyError(msg)

    try:
        return float(_Evaluator(allow_names=True, namespace=ns).visit(tree))
    except _SafeEvalError as e:
        msg = f"Invalid expression: {expr!r} ({e})"
        raise ValueError(msg) from None


# =============================================================================
# Normalization helpers used by the parser
# =============================================================================


def _collect_scalars_and_initial_specs(
    params: Mapping[str, Any],
    compartments: Iterable[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Collect scalar params and initial-specs from a unified parameter block.

    This organizes input without splitting user-facing parameters. At most one
    initial spec per compartment is allowed.

    Args:
        params: Raw parameters mapping from YAML.
        compartments: Ordered list of compartment names.

    Returns:
        A pair ``(raw_scalars, initials_spec_by_compartment)``.

    Raises:
        ValueError: If an initial spec is invalid or duplicated per compartment,
            or if an initial references an unknown compartment, or if a role is
            unsupported.
        TypeError: If a parameter is neither a scalar (int/float/str) nor an
            initial-spec mapping.
    """
    comps = list(compartments)
    scalars_raw: dict[str, Any] = {}
    initials_by_comp: dict[str, dict[str, Any]] = {c: {} for c in comps}

    for name, spec in params.items():
        if isinstance(spec, Mapping) and spec.get("role") == "initial":
            comp = spec.get("of")
            if comp not in comps:
                msg = f"Initial '{name}' references unknown compartment '{comp}'."
                raise ValueError(msg)
            if "values" not in spec and "expression" not in spec:
                msg = f"Initial '{name}' must have 'values' or 'expression'."
                raise ValueError(msg)
            if initials_by_comp[comp]:
                msg = f"Multiple initial specs for compartment '{comp}'."
                raise ValueError(msg)
            copied = dict(spec)
            copied["__param_key__"] = name
            initials_by_comp[comp] = copied
            continue

        if isinstance(spec, (int, float, str)):
            scalars_raw[name] = spec
            continue

        if isinstance(spec, Mapping) and "role" in spec:
            msg = (
                f"Parameter '{name}' has unsupported role '{spec.get('role')}'. "
                "Only role 'initial' is recognized in the unified parameters block."
            )
            raise ValueError(msg)

        msg = f"Parameter '{name}' must be a scalar (int/float/str) "
        msg += "or an initial-role mapping."
        raise TypeError(msg)

    return scalars_raw, initials_by_comp


def _normalize_scalar_parameters(
    raw_params: Mapping[str, Any],
) -> dict[str, float]:
    """Normalize scalar parameters to floats.

    Args:
        raw_params: Raw parameter mapping with int/float or numeric strings.

    Returns:
        Mapping from parameter name to float value.

    Raises:
        TypeError: If a value is not a scalar or numeric string.
    """
    out: dict[str, float] = {}
    for k, v in raw_params.items():
        if isinstance(v, (int, float, str)):
            out[k] = _safe_eval_numeric(v)
        else:
            msg = f"Parameter '{k}' must be a scalar or numeric string; got {type(v)}."
            raise TypeError(msg)
    return out


def _resolve_initials(  # noqa: C901
    initials: Mapping[str, dict[str, Any]],
    scalars: Mapping[str, float],
) -> dict[str, float]:
    """Resolve initial conditions across compartments.

    Supports:
        * values: numeric literal or numeric string.
        * expression: arithmetic string referencing scalar params or other initials.

    Expressions are evaluated in dependency order.

    Args:
        initials: Map of compartment -> spec dict with 'values' or 'expression'.
        scalars: Already-evaluated scalar parameters.

    Returns:
        Map of compartment -> resolved initial value.

    Raises:
        ValueError: If dependencies cannot be resolved or a cycle is detected.
    """
    remaining = {c: dict(spec) for c, spec in initials.items() if spec}
    resolved: dict[str, float] = {}

    # Direct values
    for comp, spec in list(remaining.items()):
        if "values" in spec and spec["values"] is not None:
            resolved[comp] = _safe_eval_numeric(spec["values"])
            del remaining[comp]

    # Expressions (dependency ordering)
    guard = 0
    while remaining:
        progressed = False
        for comp, spec in list(remaining.items()):
            expr = spec.get("expression")
            if not isinstance(expr, str):
                continue
            ns: dict[str, float] = {**scalars, **resolved}
            ns.update({f"{k}0": v for k, v in resolved.items()})
            try:
                val = _safe_eval_symbolic(expr, ns)
            except KeyError:
                continue
            resolved[comp] = float(val)
            del remaining[comp]
            progressed = True

        guard += 1
        if not progressed:
            missing = {c: remaining[c].get("expression") for c in remaining}
            msg = (
                "Could not resolve initial expressions "
                "(dependencies unresolved): "
                f"{missing}"
            )
            raise ValueError(msg)
        if guard > 100:
            msg = "Cycle detected in initial-condition expressions."
            raise ValueError(msg)

    # Default any missing compartments to zero
    for comp in initials:
        if comp not in resolved:
            resolved[comp] = 0.0

    return resolved


def _normalize_dynamics(
    rules: list[Any],
    compartments: list[str],
) -> list[dict[str, Any]]:
    """Convert dynamic rules to an index-based form, preserving rate expressions.

    Args:
        rules: Rule-like objects with name, source, target, rate, broadcast (opt).
        compartments: Ordered list of compartments.

    Returns:
        Rule dicts with keys: name, source_idx, target_idx, rate_expr, broadcast.

    Raises:
        ValueError: If a rule references an unknown compartment.
    """
    name_to_idx = {c: i for i, c in enumerate(compartments)}
    out: list[dict[str, Any]] = []
    for r in rules:
        if r.source not in name_to_idx or r.target not in name_to_idx:
            msg = f"Unknown source/target in rule '{r.name}'."
            raise ValueError(msg)
        out.append({
            "name": r.name,
            "source_idx": name_to_idx[r.source],
            "target_idx": name_to_idx[r.target],
            "rate_expr": r.rate,
            "broadcast": r.broadcast or None,
        })
    return out
