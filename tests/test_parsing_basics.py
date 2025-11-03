"""Unit tests for gempyor2.parser normalization and linting."""

from __future__ import annotations

import textwrap
import pytest

from gempyor2 import parse_model_yaml, parse_model_dict, NormalizedIR


def test_parse_minimal_sir_yaml():
    """Happy-path SIR config parses; scalars/initials/dynamics/engine are normalized."""
    yaml_text = textwrap.dedent(
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
    ir = parse_model_yaml(yaml_text)
    assert isinstance(ir, NormalizedIR)
    assert ir.model == "SIR-min"
    assert ir.t_span == (0.0, 40.0)
    assert ir.structure["compartments"] == ["S", "I", "R"]
    assert ir.parameters["beta"] == pytest.approx(0.4)
    assert ir.parameters["gamma"] == pytest.approx(0.2)
    assert ir.parameters["N"] == pytest.approx(1000.0)
    assert ir.initial_conditions == {"S": pytest.approx(995.0), "I": pytest.approx(5.0), "R": pytest.approx(0.0)}
    assert ir.y0 == [pytest.approx(995.0), pytest.approx(5.0), pytest.approx(0.0)]
    names = {d["name"] for d in ir.dynamics}
    assert names == {"infection", "recovery"}
    inf = next(d for d in ir.dynamics if d["name"] == "infection")
    rec = next(d for d in ir.dynamics if d["name"] == "recovery")
    assert (inf["source_idx"], inf["target_idx"]) == (0, 1)
    assert (rec["source_idx"], rec["target_idx"]) == (1, 2)
    assert ir.engine["module"] == "scipy.integrate.solve_ivp"
    assert ir.engine["options"]["method"] == "RK45"
    assert isinstance(ir.warnings, list)


def test_axes_validation_duplicates_raise():
    """Axes with duplicate labels fail validation."""
    yaml_text = textwrap.dedent(
        """
        model: SIR-axes
        t0: 0
        tf: 10
        structure:
          compartments: [S, I, R]
          axes:
            age: [A, A]
        parameters: {}
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    with pytest.raises(Exception):
        parse_model_yaml(yaml_text)


def test_time_bounds_invalid():
    """tf must be strictly greater than t0."""
    yaml_text = textwrap.dedent(
        """
        model: bad-time
        t0: 5
        tf: 5
        structure:
          compartments: [S, I]
        parameters: {}
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    with pytest.raises(Exception):
        parse_model_yaml(yaml_text)


def test_unknown_symbol_in_rate_lints_and_optionally_errors():
    """Unknown names in rate expressions produce lints; can be escalated to errors."""
    yaml_text = textwrap.dedent(
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
            rate: "beta * S * I / N"  # N is unknown
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    ir = parse_model_yaml(yaml_text)
    assert any("unknown symbol(s) in rate" in w for w in ir.warnings)
    with pytest.raises(ValueError):
        parse_model_yaml(yaml_text, warnings_as_errors=True)


def test_initials_dependency_resolution():
    """Initial expressions resolve in dependency order across A0, B0, C0."""
    yaml_text = textwrap.dedent(
        """
        model: deps
        t0: 0
        tf: 1
        structure:
          compartments: [A, B, C]
        parameters:
          total: 100
          A0_parm: { role: initial, of: A, values: 10 }
          B0_parm: { role: initial, of: B, expression: "A0 + 5" }
          C0_parm: { role: initial, of: C, expression: "total - A0 - B0" }
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    ir = parse_model_yaml(yaml_text)
    assert ir.initial_conditions["A"] == pytest.approx(10.0)
    assert ir.initial_conditions["B"] == pytest.approx(15.0)
    assert ir.initial_conditions["C"] == pytest.approx(75.0)
    assert ir.y0 == [pytest.approx(10.0), pytest.approx(15.0), pytest.approx(75.0)]


def test_initial_cycle_detection_errors():
    """Cycles among initial expressions raise a hard error."""
    yaml_text = textwrap.dedent(
        """
        model: cycle
        t0: 0
        tf: 1
        structure:
          compartments: [X, Y]
        parameters:
          X_init: { role: initial, of: X, expression: "Y0 + 1" }
          Y_init: { role: initial, of: Y, expression: "X0 + 1" }
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    with pytest.raises(ValueError):
        parse_model_yaml(yaml_text)


def test_duplicate_initial_for_same_compartment_errors():
    """Multiple initial specs for the same compartment are rejected."""
    yaml_text = textwrap.dedent(
        """
        model: dup
        t0: 0
        tf: 1
        structure:
          compartments: [S, I]
        parameters:
          S_init_1: { role: initial, of: S, values: 10 }
          S_init_2: { role: initial, of: S, values: 20 }
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    with pytest.raises(ValueError):
        parse_model_yaml(yaml_text)


def test_initial_references_unknown_compartment_errors():
    """Initial spec referencing an unknown compartment raises an error."""
    yaml_text = textwrap.dedent(
        """
        model: bad-init
        t0: 0
        tf: 1
        structure:
          compartments: [S]
        parameters:
          bad: { role: initial, of: I, values: 1 }  # I not in structure
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    with pytest.raises(ValueError):
        parse_model_yaml(yaml_text)


def test_scalar_normalization_accepts_numeric_strings_and_exprs():
    """Scalar parameters accept numeric strings/expressions and normalize to floats."""
    yaml_text = textwrap.dedent(
        """
        model: scalars
        t0: 0
        tf: 1
        structure:
          compartments: [S]
        parameters:
          a: "1/5"
          b: "2**3"
          c: 0.25
          S0: { role: initial, of: S, values: 1 }
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    ir = parse_model_yaml(yaml_text)
    assert ir.parameters["a"] == pytest.approx(0.2)
    assert ir.parameters["b"] == pytest.approx(8.0)
    assert ir.parameters["c"] == pytest.approx(0.25)


def test_parse_model_dict_and_y0_defaults_missing_initials_to_zero():
    """parse_model_dict works; missing initials default to zero for y0 assembly."""
    cfg = {
        "model": "defaults",
        "t0": 0,
        "tf": 1,
        "structure": {"compartments": ["A", "B"]},
        "parameters": {
            "alpha": 1.0,
            "A_init": {"role": "initial", "of": "A", "values": 2},
            # B init omitted -> defaults to 0
        },
        "dynamics": [],
        "engine": {"module": "scipy.integrate.solve_ivp", "options": {}},
    }
    ir = parse_model_dict(cfg)
    assert ir.parameters["alpha"] == pytest.approx(1.0)
    assert ir.initial_conditions["A"] == pytest.approx(2.0)
    assert ir.initial_conditions["B"] == pytest.approx(0.0)
    assert ir.y0 == [pytest.approx(2.0), pytest.approx(0.0)]


def test_rule_references_unknown_compartment_errors():
    """Rules referencing unknown compartments raise a hard error during normalization."""
    yaml_text = textwrap.dedent(
        """
        model: bad-rule
        t0: 0
        tf: 1
        structure:
          compartments: [S, I]
        parameters:
          S_init: { role: initial, of: S, values: 1 }
          I_init: { role: initial, of: I, values: 0 }
        dynamics:
          - name: bad
            source: S
            target: R   # R is not defined
            rate: "1.0"
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    with pytest.raises(ValueError):
        parse_model_yaml(yaml_text)


def test_initial_expr_may_not_reference_time_varying_states():
    """Initial expressions cannot reference state variables (S, I, etc.)."""
    yaml_text = textwrap.dedent(
        """
        model: bad-init-state
        t0: 0
        tf: 1
        structure:
          compartments: [S, I]
        parameters:
          S_init: { role: initial, of: S, expression: "I + 1" }  # illegal
          I_init: { role: initial, of: I, values: 0 }
        dynamics: []
        engine:
          module: "scipy.integrate.solve_ivp"
          options: {}
        """
    )
    ir = parse_model_yaml(yaml_text)
    assert any("time-varying state(s)" in w for w in ir.warnings)
    with pytest.raises(ValueError):
        parse_model_yaml(yaml_text, warnings_as_errors=True)
