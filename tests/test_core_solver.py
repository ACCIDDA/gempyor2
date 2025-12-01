# tests/test_core_solver.py
"""Unit tests for the CoreSolver class.

This module contains tests that verify:
- CoreSolver operates correctly in Euler mode without operators.
- CoreSolver with Crank-Nicolson operators matches direct implicit solve.
- CoreSolver with predictor-corrector operators handles shapes correctly.
- CoreSolver.run_imex implements a Heun-style IMEX step correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from gempyor2.core_solver import CoreSolver
from gempyor2.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
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

    def rhs_func(
        t: float,  # noqa: ARG001
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
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
    dt = 1.0  # match the single time step for simplicity

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    left, right = build_crank_nicolson_dense(geom, cfg, dt)

    def rhs_func(
        t: float,  # noqa: ARG001
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        # CN is applied to this rhs; keep it equal to current state
        return state

    solver = CoreSolver(core, operators=(left, right))

    # Do one step by hand
    x0_flat = x0[:, 0]
    x1_manual_flat = implicit_solve(left, right, x0_flat)
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

    # Simple A for predictor-corrector, with explicit dt scaling:
    dx = 1.0
    coeff = 0.05
    dt = float(core.dt)

    lap = build_laplacian_tridiag(
        n=n,
        dx=dx,
        coeff=coeff,
        dtype=np.float64,
        bc="neumann",
    ).toarray()
    time_scaled_op = dt * lap
    predictor, left, right = build_predictor_corrector(time_scaled_op)

    def rhs_func(
        t: float,  # noqa: ARG001
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        # Just return state to be processed by predictor + implicit solve
        return state

    solver = CoreSolver(core, operators=(predictor, left, right))
    solver.run(rhs_func)

    # Just check we advanced correctly and have finite values
    assert core.current_step == 1
    x1 = core.get_state_at(1)
    assert x1.shape == (n, 2)
    assert np.all(np.isfinite(x1))


def test_core_solver_run_imex_constant_reaction_identity_operators() -> None:
    """Test run_imex with identity operators and constant reaction.

    We consider the simple ODE

        y' = 1

    with initial condition y(0) = 0. Using identity operators L = R = I
    means the implicit_solve is effectively the identity, so run_imex should
    reduce to a pure Heun (trapezoidal) step for the reaction term.

    For a constant reaction F(t, y) = 1, Heun is exact, so after total time T
    we should have y(T) = T.
    """
    # Time grid: two steps of dt = 0.5, total T = 1.0
    time_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    core = ModelCore(
        n_states=1,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=True,
    )

    init = np.array([[0.0]], dtype=float)
    core.set_initial_state(init)

    # Identity operators: L = I, R = I
    n = 1
    identity_mat = np.eye(n, dtype=float)
    left = identity_mat.copy()
    right = identity_mat.copy()

    def reaction_rhs(
        t: float,  # noqa: ARG001
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        # Constant reaction: F(t, y) = 1
        return np.ones_like(state)

    solver = CoreSolver(core, operators=(left, right))
    solver.run_imex(reaction_rhs)

    # After T = 1.0 with y' = 1 and y(0) = 0, exact solution is y(T) = 1.
    final_state = core.current_state
    assert final_state.shape == (1, 1)
    assert np.allclose(final_state[0, 0], 1.0, atol=1e-12, rtol=0.0)
