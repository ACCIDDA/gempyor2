# tests/test_core_solver.py
"""Unit tests for the CoreSolver class.

This module contains tests that verify:
- CoreSolver operates correctly in Euler mode without operators
- CoreSolver with Crank-Nicolson operators matches direct implicit solve
- CoreSolver with predictor-corrector operators handles shapes correctly
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gempyor2.core_solver import CoreSolver
from gempyor2.matrix_ops import (
    build_crank_nicolson_dense,
    build_laplacian_tridiag,
    build_predictor_corrector,
    implicit_solve,
)
from gempyor2.model_core import ModelCore


def test_core_solver_euler_mode_uses_rhs_as_next_state() -> None:
    """Test CoreSolver in Euler-like mode (no operators)."""
    time_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    core = ModelCore(
        n_states=1,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=True,
    )

    init = np.array([[0.0]], dtype=float)
    core.set_initial_state(init)

    def rhs_func(t: float, state: NDArray[np.floating]) -> NDArray[np.floating]:  # noqa: ARG001
        # Next state increments by 1 each step (Euler-like mode)
        return state + 1.0

    solver = CoreSolver(core, operators=None)
    solver.run(rhs_func)

    # Expect: step 0 -> 0, step 1 -> 1, step 2 -> 2, step 3 -> 3
    expected = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float).reshape(-1, 1, 1)
    # state_array shape: (n_timesteps, n_states, n_subgroups)
    assert core.state_array is not None
    assert np.allclose(core.state_array[:, 0, 0], expected[:, 0, 0])


def test_core_solver_cn_matches_direct_implicit_solve_single_group() -> None:
    """Test CoreSolver with Crank-Nicolson operators matches direct implicit_solve."""
    # Single group, CN operators, one time step
    time_grid = np.array([0.0, 1.0], dtype=float)
    n = 5
    core = ModelCore(
        n_states=n,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=True,
    )

    # Initial state: some smooth pattern
    x0 = np.linspace(0.0, 1.0, n, dtype=float).reshape(n, 1)
    core.set_initial_state(x0)

    dx = 1.0
    coeff = 0.1
    L, R = build_crank_nicolson_dense(n, dx, coeff)

    def rhs_func(t: float, state: NDArray[np.floating]) -> NDArray[np.floating]:  # noqa: ARG001
        # CN is applied to this rhs; keep it equal to current state
        return state

    solver = CoreSolver(core, operators=(L, R))

    # Do one step by hand
    x0_flat = x0[:, 0]
    x1_manual_flat = implicit_solve(L, R, x0_flat)
    x1_manual = x1_manual_flat.reshape(n, 1)

    solver.run(rhs_func)

    # There are 2 timesteps; after run, current_step should be 1
    assert core.current_step == 1
    x1_core = core.get_state_at(1)

    assert np.allclose(x1_core, x1_manual)


def test_core_solver_predictor_corrector_path_shapes() -> None:
    """Test CoreSolver with predictor-corrector operators runs without shape issues."""
    # Sanity test that predictor-corrector path runs without shape issues.
    time_grid = np.array([0.0, 1.0], dtype=float)
    n = 4
    core = ModelCore(
        n_states=n,
        n_subgroups=2,
        time_grid=time_grid,
        store_history=True,
    )

    init = np.ones((n, 2), dtype=float)
    core.set_initial_state(init)

    # Simple "A" for predictor-corrector:
    dx = 1.0
    coeff = 0.05
    A = build_laplacian_tridiag(n, dx, coeff)
    predictor, L, R = build_predictor_corrector(A)

    def rhs_func(t: float, state: NDArray[np.floating]) -> NDArray[np.floating]:  # noqa: ARG001
        # Just return state to be processed by predictor+CN
        return state

    solver = CoreSolver(core, operators=(predictor, L, R))
    solver.run(rhs_func)

    # Just check we advanced correctly and have finite values
    assert core.current_step == 1
    x1 = core.get_state_at(1)
    assert x1.shape == (n, 2)
    assert np.all(np.isfinite(x1))
