# src/gempyor2/flepi_model.py
"""High-level Pydantic wrapper that orchestrates parsing, linting, and normalization.

This object keeps the low-level parser intact while providing a single place to:
- hold the validated ModelSpec,
- enforce explicit choice of `warnings_as_errors`,
- expose the computed symbol table and lint warnings,
- expose the fully normalized IR (ready for integration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml as _yaml  # types provided by types-PyYAML in dev deps

if TYPE_CHECKING:
    from collections.abc import Mapping

from pydantic import BaseModel, Field, model_validator

from ._utils import (
    _collect_scalars_and_initial_specs,
    _normalize_dynamics,
    _normalize_scalar_parameters,
    _resolve_initials,
)
from .parser import (
    ModelSpec,
    NormalizedIR,
    build_symbol_table,
    lint_expressions,
    parse_model_dict,
    parse_model_yaml,
)


class FlepiModel(BaseModel):
    """Meta wrapper around the parsing/normalization pipeline."""

    # Required: caller must make an explicit choice.
    warnings_as_errors: bool

    # The validated spec well run through the pipeline.
    spec: ModelSpec

    # Computed artifacts, populated by the model validator.
    symbols: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    ir: NormalizedIR | None = None

    model_config = {
        # We store dataclasses and non-pydantic objects
        # (NormalizedIR, SymbolInfo entries).
        "arbitrary_types_allowed": True
    }

    @model_validator(mode="after")
    def _build_pipeline(self) -> FlepiModel:
        """Run the same pipeline as normalize_spec_to_ir, but also keep symbols.

        Returns:
            The validated and fully-initialized `FlepiModel` instance (self).
        """
        comps = self.spec.structure.compartments

        # 1) Gather and normalize parameters / initials
        raw_scalars, initials_raw = _collect_scalars_and_initial_specs(
            self.spec.parameters, comps
        )
        scalars = _normalize_scalar_parameters(raw_scalars)

        # 2) Symbols + lint (honor `warnings_as_errors`)
        symbols = build_symbol_table(self.spec, scalars, initials_raw)
        warnings = lint_expressions(
            self.spec,
            symbols,
            initials_raw,
            warnings_as_errors=self.warnings_as_errors,
        )

        # 3) Resolve initials (dependency ordered), dynamics, engine, structure
        initials = _resolve_initials(initials_raw, scalars)
        y0 = [float(initials.get(c, 0.0)) for c in comps]
        dynamics = _normalize_dynamics(self.spec.dynamics, comps)
        engine = {
            "module": self.spec.engine.module,
            "options": self.spec.engine.options,
        }
        t_span: tuple[float, float] = (float(self.spec.t0), float(self.spec.tf))

        structure: dict[str, Any] = {"compartments": comps}
        if self.spec.structure.axes:
            structure["axes"] = self.spec.structure.axes

        # 4) Assemble IR
        ir = NormalizedIR(
            model=self.spec.model,
            t_span=t_span,
            structure=structure,
            parameters=scalars,
            initial_conditions=initials,
            y0=y0,
            dynamics=dynamics,
            engine=engine,
            warnings=warnings,
        )

        # Persist artifacts on the instance
        self.symbols = symbols
        self.warnings = warnings
        self.ir = ir
        return self

    # --------------------------
    # Convenience constructors
    # --------------------------
    @classmethod
    def from_yaml(
        cls, yaml_str: str, *, warnings_as_errors: bool = False
    ) -> FlepiModel:
        """Build from a YAML string.

        Returns:
            A fully-initialized `FlepiModel` created from the YAML text.
        """
        # Keep parser logic authoritative for spec validation:

        # Parse once to ensure behavior matches `parse_model_yaml`:
        _ = parse_model_yaml(yaml_str, warnings_as_errors=warnings_as_errors)
        # Recreate spec to keep both spec + ir available in the wrapper:
        data = _yaml.safe_load(yaml_str)
        spec = ModelSpec.model_validate(data)
        return cls(spec=spec, warnings_as_errors=warnings_as_errors)

    @classmethod
    def from_dict(
        cls,
        cfg: Mapping[str, Any],
        *,
        warnings_as_errors: bool = False,
    ) -> FlepiModel:
        """Build from an in-memory config mapping.

        Returns:
            A fully-initialized `FlepiModel` created from the provided mapping.
        """
        _ = parse_model_dict(dict(cfg), warnings_as_errors=warnings_as_errors)
        spec = ModelSpec.model_validate(dict(cfg))
        return cls(spec=spec, warnings_as_errors=warnings_as_errors)

    # --------------------------
    # Convenience accessors
    # --------------------------
    @property
    def model(self) -> str:
        """Return the model name from the normalized IR or spec."""
        return self.ir.model if self.ir else self.spec.model

    @property
    def t_span(self) -> tuple[float, float]:
        """Return the time span (t0, tf) from the normalized IR or spec."""
        return self.ir.t_span if self.ir else (float(self.spec.t0), float(self.spec.tf))

    @property
    def compartments(self) -> list[str]:
        """Return the list of compartment names from the model structure."""
        return list(self.spec.structure.compartments)

    @property
    def y0(self) -> list[float]:
        """Return the initial conditions vector (y0) from the normalized IR."""
        return list(self.ir.y0) if self.ir else []

    @property
    def parameters(self) -> dict[str, float]:
        """Return the normalized scalar parameters from the IR."""
        return dict(self.ir.parameters) if self.ir else {}

    @property
    def dynamics(self) -> list[dict[str, Any]]:
        """Return the list of normalized dynamics specifications."""
        return list(self.ir.dynamics) if self.ir else []

    @property
    def engine(self) -> dict[str, Any]:
        """Return the engine configuration dictionary."""
        return dict(self.ir.engine) if self.ir else {}
