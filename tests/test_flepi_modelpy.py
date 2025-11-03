# tests/test_flepi_model.py
"""Tests for the FlepiModel class and its integration with the parser.

This module contains unit tests for:
- FlepiModel.from_yaml and FlepiModel.from_dict constructors
- IR structure validation and parity with parse_model_yaml
- Warning and error handling (warnings_as_errors mode)
- Symbol table construction including scalars, initial params, and aliases
- Axes passthrough in model structure
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from gempyor2.flepi_model import FlepiModel
from gempyor2.parser import NormalizedIR, Role, parse_model_yaml


def _sir_yaml() -> str:
    return textwrap.dedent(
        """
        model: SIR-min
        t0: 0
        tf: 40
        structure:
          compartments: [S, I, R]
        parameters:
          beta: "0.4"
          gamma: 0.2
          N: 1000
          S_init:
            role: initial
            of: S
            expression: "N - I0 - R0"
          I_init:
            role: initial
            of: I
            values: 5
          R_init:
            role: initial
            of: R
            values: 0
        dynamics:
          - name: infection
            source: S
            target: I
            rate: "beta * S * I / N"
          - name: recovery
            source: I
            target: R
            rate: "gamma * I"
        engine:
          module: "scipy.integrate.solve_ivp"
          options:
            method: "RK45"
        """
    )


def test_from_yaml_builds_and_matches_parser_ir() -> None:
    """FlepiModel.from_yaml runs the pipeline and matches parse_model_yaml.

    Core fields are validated for parity.
    """
    yml = _sir_yaml()

    # Ground truth IR from parser
    ir_ref: NormalizedIR = parse_model_yaml(yml)

    # Wrapper
    fm = FlepiModel.from_yaml(yml, warnings_as_errors=False)

    assert fm.ir is not None
    ir = fm.ir

    # Parity checks (spot-check main surfaces)
    assert ir.model == ir_ref.model
    assert ir.t_span == ir_ref.t_span
    assert ir.structure == ir_ref.structure
    assert ir.parameters == ir_ref.parameters
    assert ir.initial_conditions == ir_ref.initial_conditions
    assert ir.y0 == ir_ref.y0
    assert [
        {k: d[k] for k in ("name", "source_idx", "target_idx", "rate_expr")}
        for d in ir.dynamics
    ] == [
        {k: d[k] for k in ("name", "source_idx", "target_idx", "rate_expr")}
        for d in ir_ref.dynamics
    ]
    assert ir.engine == ir_ref.engine

    # Accessors
    assert fm.model == ir_ref.model
    assert fm.t_span == ir_ref.t_span
    assert fm.compartments == ir_ref.structure["compartments"]
    assert fm.y0 == ir_ref.y0
    assert fm.parameters == ir_ref.parameters
    assert fm.dynamics == ir_ref.dynamics
    assert fm.engine == ir_ref.engine


def test_from_dict_equivalent_to_from_yaml() -> None:
    """from_dict reproduces the same result as from_yaml when given the same config."""
    yml = _sir_yaml()
    cfg = yaml.safe_load(yml)

    fm_yaml = FlepiModel.from_yaml(yml, warnings_as_errors=False)
    fm_dict = FlepiModel.from_dict(cfg, warnings_as_errors=False)

    assert fm_yaml.ir is not None
    assert fm_dict.ir is not None
    assert fm_yaml.ir.model == fm_dict.ir.model
    assert fm_yaml.ir.t_span == fm_dict.ir.t_span
    assert fm_yaml.ir.structure == fm_dict.ir.structure
    assert fm_yaml.ir.parameters == fm_dict.ir.parameters
    assert fm_yaml.ir.initial_conditions == fm_dict.ir.initial_conditions
    assert fm_yaml.ir.y0 == fm_dict.ir.y0
    assert fm_yaml.ir.dynamics == fm_dict.ir.dynamics
    assert fm_yaml.ir.engine == fm_dict.ir.engine


def test_warnings_as_errors_is_enforced() -> None:
    """A lint in rates should raise when warnings_as_errors=True."""
    yml = textwrap.dedent(
        """
        model: lint-rate
        t0: 0
        tf: 1
        structure:
          compartments: [S, I]
        parameters:
          beta: 0.5
          S_init: { role: initial, of: S, values: 10 }
          I_init: { role: initial, of: I, values: 1 }
        dynamics:
          - name: infect
            source: S
            target: I
            rate: "beta * S * I / N"  # N is unknown -> lint
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )

    # No error when not strict; warnings should be present on the wrapper.
    fm = FlepiModel.from_yaml(yml, warnings_as_errors=False)
    assert any("unknown symbol(s) in rate" in w for w in fm.warnings)

    # Error when strict
    with pytest.raises(ValueError, match="unknown symbol"):
        FlepiModel.from_yaml(yml, warnings_as_errors=True)


def test_symbol_table_contains_scalars_and_initial_aliases() -> None:
    """Symbol table includes scalar params, initial param keys, and {comp}0 aliases."""
    yml = _sir_yaml()
    fm = FlepiModel.from_yaml(yml, warnings_as_errors=False)

    # Required symbol names
    for name in ("beta", "gamma", "N", "S_init", "I_init", "R_init", "S0", "I0", "R0"):
        assert name in fm.symbols, f"expected symbol {name!r} in symbol table"

    # Optional: roles, when available (Role from parser)
    # Scalars
    assert fm.symbols["beta"].role == Role.scalar
    assert fm.symbols["gamma"].role == Role.scalar
    assert fm.symbols["N"].role == Role.scalar
    # Initial param keys
    assert fm.symbols["S_init"].role == Role.initial
    assert fm.symbols["I_init"].role == Role.initial
    assert fm.symbols["R_init"].role == Role.initial
    # Aliases
    assert fm.symbols["S0"].role == Role.initial
    assert fm.symbols["I0"].role == Role.initial
    assert fm.symbols["R0"].role == Role.initial


def test_axes_passthrough_in_structure() -> None:
    """Axes present in spec are preserved in the IR structure."""
    yml = textwrap.dedent(
        """
        model: axes-demo
        t0: 0
        tf: 5
        structure:
          compartments: [S, I]
          axes:
            age: [A, B, C]
        parameters:
          S_init: { role: initial, of: S, values: 10 }
          I_init: { role: initial, of: I, values: 1 }
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    fm = FlepiModel.from_yaml(yml, warnings_as_errors=False)
    assert fm.ir is not None
    ir = fm.ir
    assert "axes" in ir.structure
    assert ir.structure["axes"] == {"age": ["A", "B", "C"]}
