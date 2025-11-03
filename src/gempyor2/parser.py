# src/gempyor2/parser.py
"""Pydantic models and normalization pipeline for gempyor2 parser."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum  # UP042: use StrEnum instead of (str, Enum)
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ._utils import (
    _collect_scalars_and_initial_specs,
    _names_in_expr,
    _normalize_dynamics,
    _normalize_scalar_parameters,
    _resolve_initials,
)

# =============================================================================
# Pydantic models
# =============================================================================


class EngineConfig(BaseModel):
    """Engine configuration for the solver module."""

    module: str
    options: dict[str, Any] = Field(default_factory=dict)


class DynamicRule(BaseModel):
    """A single transition rule between compartments."""

    name: str
    source: str
    target: str
    rate: str
    broadcast: list[str] | None = None

    @field_validator("rate")
    @classmethod
    def _non_empty_rate(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            msg = "rate must be a non-empty string expression."
            raise ValueError(msg)
        return v


class Structure(BaseModel):
    """Model structure: compartments and optional axes."""

    compartments: list[str]
    axes: dict[str, list[str]] | None = None

    @field_validator("compartments")
    @classmethod
    def _validate_compartments(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "structure.compartments must be non-empty."
            raise ValueError(msg)
        if len(v) != len(set(v)):
            msg = "structure.compartments must be unique."
            raise ValueError(msg)
        return v

    @field_validator("axes")
    @classmethod
    def _validate_axes(
        cls, axes: dict[str, list[str]] | None
    ) -> dict[str, list[str]] | None:
        if axes is None:
            return None
        if not isinstance(axes, dict):
            msg = "structure.axes must be a mapping of axis_name -> list[str]."
            raise TypeError(msg)
        for ax, labels in axes.items():
            if not isinstance(labels, list) or not all(
                isinstance(x, str) for x in labels
            ):
                msg = f"Axis '{ax}' must be a list of strings."
                raise TypeError(msg)
            if not labels:
                msg = f"Axis '{ax}' must not be empty."
                raise ValueError(msg)
            if len(labels) != len(set(labels)):
                msg = f"Axis '{ax}' labels must be unique."
                raise ValueError(msg)
        return axes


class ModelSpec(BaseModel):
    """Top-level model specification from YAML."""

    model: str
    t0: float
    tf: float
    structure: Structure
    parameters: dict[str, Any]
    dynamics: list[DynamicRule]
    engine: EngineConfig

    @model_validator(mode="after")
    def _validate_times(self) -> ModelSpec:
        if float(self.tf) <= float(self.t0):
            msg = "tf must be greater than t0."
            raise ValueError(msg)
        return self


# =============================================================================
# Roles & Symbol table
# =============================================================================


class Role(StrEnum):
    """Semantic role for a named quantity (scalar/initial/derived)."""

    scalar = "scalar"
    initial = "initial"
    derived = "derived"


@dataclass
class SymbolInfo:
    """Metadata for a symbol in the unified namespace."""

    name: str
    role: Role
    of: str | None = None
    shape: tuple[int, ...] | None = None
    has_prior: bool = False
    fixed: bool = False


# =============================================================================
# Normalized IR
# =============================================================================


@dataclass
class NormalizedIR:
    """Normalized intermediate representation ready for integration."""

    model: str
    t_span: tuple[float, float]
    structure: dict[str, Any]
    parameters: dict[str, float]
    initial_conditions: dict[str, float]
    y0: list[float]
    dynamics: list[dict[str, Any]]
    engine: dict[str, Any]
    warnings: list[str]


def build_symbol_table(
    _spec: ModelSpec,
    scalar_values: dict[str, float],
    initials_spec: dict[str, dict[str, Any]],
) -> dict[str, SymbolInfo]:
    """Create a unified symbol table (parameters + initials).

    Args:
        _spec: The validated model specification (reserved for future use).
        scalar_values: Normalized scalar parameter values.
        initials_spec: Initial specification per compartment.

    Returns:
        Mapping from symbol name to its metadata.
    """
    symbols: dict[str, SymbolInfo] = {}

    for name in scalar_values:
        symbols[name] = SymbolInfo(name=name, role=Role.scalar, fixed=True)

    for comp, init in initials_spec.items():
        if not init:
            continue
        param_key = init.get("__param_key__")
        alias = f"{comp}0"
        if param_key:
            symbols[param_key] = SymbolInfo(name=param_key, role=Role.initial, of=comp)
        symbols.setdefault(alias, SymbolInfo(name=alias, role=Role.initial, of=comp))
    return symbols


def lint_expressions(
    spec: ModelSpec,
    symbols: dict[str, SymbolInfo],
    initials_spec: dict[str, dict[str, Any]],
    *,
    warnings_as_errors: bool = False,
) -> list[str]:
    """Lint rate and initial expressions for conceptual issues; return warnings.

    Args:
        spec: The validated model specification.
        symbols: The unified symbol table from ``build_symbol_table``.
        initials_spec: Initial specification per compartment.
        warnings_as_errors: If True, raise on any lint warnings.

    Returns:
        A list of human-readable warning strings.

    Raises:
        ValueError: If ``warnings_as_errors`` is True and any lint warnings exist.
    """
    warnings: list[str] = []

    comp_names = set(spec.structure.compartments)
    initial_aliases = {f"{c}0" for c in comp_names} | {
        init.get("__param_key__") for init in initials_spec.values() if init
    }
    initial_aliases.discard(None)

    known_names_for_rates = set(symbols.keys()) | comp_names

    for rule in spec.dynamics:
        names = _names_in_expr(rule.rate)
        bad_initials = {n for n in names if n in initial_aliases}
        if bad_initials:
            sug = sorted(n[:-1] for n in bad_initials if n.endswith("0"))
            msg = (
                f"[lint] Rule '{rule.name}': rate uses symbols {sorted(bad_initials)}; "
                f"did you mean state variables {sug}?"
            )
            warnings.append(msg)
        unknown = {n for n in names if n not in known_names_for_rates}
        if unknown:
            msg = (
                f"[lint] Rule '{rule.name}': unknown symbol(s) in rate: "
                f"{sorted(unknown)}."
            )
            warnings.append(msg)

    for comp, init in ((c, i) for c, i in initials_spec.items() if i):
        expr = init.get("expression")
        if not isinstance(expr, str):
            continue
        names = _names_in_expr(expr)
        illegal = names & comp_names
        if illegal:
            msg = (
                f"[lint] Initial '{comp}0' expression uses time-varying state(s) "
                f"{sorted(illegal)}; only scalars and other initials are allowed."
            )
            warnings.append(msg)
        allowed = set(symbols.keys()) | {f"{c}0" for c in comp_names}
        unknown = {n for n in names if n not in allowed}
        if unknown:
            msg = (
                f"[lint] Initial '{comp}0' expression has unknown name(s): "
                f"{sorted(unknown)}."
            )
            warnings.append(msg)

    if warnings_as_errors and warnings:
        msg = "Configuration lints:\n" + "\n".join(warnings)
        raise ValueError(msg)
    return warnings


def normalize_spec_to_ir(
    spec: ModelSpec,
    *,
    warnings_as_errors: bool = False,
) -> NormalizedIR:
    """Normalize a validated ModelSpec into a solver-ready IR.

    Args:
        spec: The validated model specification.
        warnings_as_errors: If True, lint warnings are escalated to an exception.

    Returns:
        The normalized intermediate representation.
    """
    warnings: list[str] = []
    comps = spec.structure.compartments

    raw_scalars, initials_raw = _collect_scalars_and_initial_specs(
        spec.parameters, comps
    )
    scalars = _normalize_scalar_parameters(raw_scalars)

    symbols = build_symbol_table(spec, scalars, initials_raw)

    warnings.extend(
        lint_expressions(
            spec,
            symbols,
            initials_raw,
            warnings_as_errors=warnings_as_errors,
        )
    )

    initials = _resolve_initials(initials_raw, scalars)
    y0 = [float(initials.get(c, 0.0)) for c in comps]
    dynamics = _normalize_dynamics(spec.dynamics, comps)

    engine = {"module": spec.engine.module, "options": spec.engine.options}
    t_span: tuple[float, float] = (float(spec.t0), float(spec.tf))

    structure = {"compartments": comps}
    if spec.structure.axes:
        structure["axes"] = spec.structure.axes

    return NormalizedIR(
        model=spec.model,
        t_span=t_span,
        structure=structure,
        parameters=scalars,
        initial_conditions=initials,
        y0=y0,
        dynamics=dynamics,
        engine=engine,
        warnings=warnings,
    )


# =============================================================================
# Public API
# =============================================================================


def parse_model_yaml(
    yaml_str: str, *, warnings_as_errors: bool = False
) -> NormalizedIR:
    """Parse and normalize a model YAML string to a NormalizedIR.

    Args:
        yaml_str: The YAML string containing the model specification.
        warnings_as_errors: If True, lint warnings are escalated to ValueError.

    Returns:
        The normalized intermediate representation parsed from YAML.

    Raises:
        TypeError: If the top-level YAML does not parse to a mapping.
    """
    data = yaml.safe_load(yaml_str)
    if not isinstance(data, dict):
        msg = "Top-level YAML must be a mapping (dict)."
        raise TypeError(msg)
    spec = ModelSpec.model_validate(data)
    return normalize_spec_to_ir(spec, warnings_as_errors=warnings_as_errors)


def parse_model_dict(
    cfg: dict[str, Any], *, warnings_as_errors: bool = False
) -> NormalizedIR:
    """Parse and normalize a model configuration dict to a NormalizedIR.

    Args:
        cfg: The configuration mapping loaded from YAML/JSON/etc.
        warnings_as_errors: If True, lint warnings are escalated to ValueError.

    Returns:
        The normalized intermediate representation parsed from the dict.

    Raises:
        TypeError: If ``cfg`` is not a mapping.
    """
    if not isinstance(cfg, dict):
        msg = "Expected a mapping for configuration."
        raise TypeError(msg)
    spec = ModelSpec.model_validate(cfg)
    return normalize_spec_to_ir(spec, warnings_as_errors=warnings_as_errors)
