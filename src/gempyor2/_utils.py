# _utils.py
from __future__ import annotations

import ast
from typing import Any

_ALLOWED_AST_NODES_NUMERIC = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Load,
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
    ast.Expr,
}

_ALLOWED_AST_NODES_SYMBOLIC = _ALLOWED_AST_NODES_NUMERIC | {ast.Name}


def _safe_eval_numeric(expr: str | float) -> float:
    """Safely evaluate a numeric expression; disallows names and function calls."""
    if isinstance(expr, (int, float)):
        return float(expr)
    if not isinstance(expr, str):
        msg = f"Expected str|int|float, got {type(expr)}"
        raise TypeError(msg)

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        t = type(node)
        if t not in _ALLOWED_AST_NODES_NUMERIC:
            msg = f"Unsupported token in numeric expression: {t.__name__}"
            raise ValueError(msg)
        if isinstance(node, ast.Name):
            msg = f"Unexpected variable '{node.id}' in numeric expression: {expr}"
            # Treat wrong value “kind” in a numeric-only expression as a TypeError
            raise TypeError(msg)

    # We compile a vetted AST and disable builtins;
    # literal_eval doesn't handle arithmetic.
    return float(eval(compile(tree, "<numeric>", "eval"), {"__builtins__": {}}, {}))  # noqa: S307


def _safe_eval_symbolic(expr: str, ns: dict[str, float]) -> float:
    """Safely evaluate arithmetic with variables present in `ns`."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        t = type(node)
        if t not in _ALLOWED_AST_NODES_SYMBOLIC:
            msg = f"Unsupported token in expression: {t.__name__}"
            raise ValueError(msg)
        if isinstance(node, ast.Call):
            msg = "Function calls are not allowed in expressions."
            # Not a type mismatch per se, but prefer TypeError per TRY004 guidance.
            raise TypeError(msg)
        if isinstance(node, ast.Name) and node.id not in ns:
            msg = f"Unknown name '{node.id}' in expression: {expr}"
            raise KeyError(msg)
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, dict(ns)))  # noqa: S307


def _names_in_expr(expr: str) -> set[str]:
    """Return all variable names used in an expression."""
    names: set[str] = set()
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _collect_scalars_and_initial_specs(
    params: dict[str, Any],
    compartments: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Walk the unified parameter block once and collect scalars and initial specs."""
    scalars_raw: dict[str, Any] = {}
    initials_by_comp: dict[str, dict[str, Any]] = {c: {} for c in compartments}

    for name, spec in params.items():
        if isinstance(spec, dict) and spec.get("role") == "initial":
            comp = spec.get("of")
            if comp not in compartments:
                msg = f"Initial '{name}' references unknown compartment '{comp}'."
                raise ValueError(msg)
            if "values" not in spec and "expression" not in spec:
                msg = "Initial '{name}' must have 'values' or 'expression'."
                raise ValueError(msg)
            if initials_by_comp[comp]:
                msg = f"Multiple initial specs for compartment '{comp}'."
                raise ValueError(msg)
            copied = dict(spec)
            copied["__param_key__"] = name
            initials_by_comp[comp] = copied
        elif isinstance(spec, (int, float, str)):
            scalars_raw[name] = spec
        elif isinstance(spec, dict) and "role" in spec:
            msg = (
                f"Parameter '{name}' has unsupported role '{spec.get('role')}'. "
                "Only role: initial is recognized in the unified parameters block."
            )
            raise ValueError(msg)
        else:
            msg = (
                f"Parameter '{name}' must be a scalar (int/float/str) "
                "or an initial-role dict."
            )
            raise TypeError(msg)

    return scalars_raw, initials_by_comp


def _normalize_scalar_parameters(raw_params: dict[str, Any]) -> dict[str, float]:
    """Normalize scalar parameters to floats."""
    out: dict[str, float] = {}
    for k, v in raw_params.items():
        if isinstance(v, (int, float, str)):
            out[k] = _safe_eval_numeric(v)
        else:
            msg = f"Parameter '{k}' must be a scalar or numeric string; got {type(v)}."
            raise TypeError(msg)
    return out


def _resolve_initials(  # noqa: C901, PLR0912 - structured loop to resolve dependencies
    initials: dict[str, dict[str, Any]],
    scalars: dict[str, float],
) -> dict[str, float]:
    """Resolve initial conditions across compartments (values or expressions)."""
    remaining = {c: spec for c, spec in initials.items() if spec}
    resolved: dict[str, float] = {}

    # Direct values
    to_delete: list[str] = []
    for comp, spec in remaining.items():
        if "values" in spec and spec["values"] is not None:
            resolved[comp] = _safe_eval_numeric(spec["values"])
            to_delete.append(comp)
    for comp in to_delete:
        del remaining[comp]

    # Expressions (dependency ordering)
    guard = 0
    while remaining:
        progressed = False
        to_delete = []
        for comp, spec in remaining.items():
            expr = spec.get("expression")
            if not isinstance(expr, str):
                continue
            ns = {**scalars}
            ns.update({f"{k}0": v for k, v in resolved.items()})
            ns.update(resolved)
            try:
                val = _safe_eval_symbolic(expr, ns)
            except KeyError:
                continue
            resolved[comp] = float(val)
            to_delete.append(comp)
            progressed = True

        for comp in to_delete:
            del remaining[comp]

        guard += 1
        if not progressed:
            if remaining:
                missing = {c: remaining[c].get("expression") for c in remaining}
                msg = (
                    "Could not resolve initial expressions (deps unresolved): "
                    f"{missing}"
                )
                raise ValueError(msg)
            break
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
    """Convert dynamic rules to an index-based form, preserving rate expressions."""
    name_to_idx = {c: i for i, c in enumerate(compartments)}
    out: list[dict[str, Any]] = []
    for r in rules:
        if r.source not in name_to_idx or r.target not in name_to_idx:
            msg = f"Unknown source/target in rule '{r.name}'."
            raise ValueError(msg)
        out.append(
            {
                "name": r.name,
                "source_idx": name_to_idx[r.source],
                "target_idx": name_to_idx[r.target],
                "rate_expr": r.rate,
                "broadcast": r.broadcast or None,
            }
        )
    return out
