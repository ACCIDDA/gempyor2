# tests/test_reaction_diffusion_imex.py
"""Tests for IMEX predictor-corrector reaction-diffusion stepping."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

from gempyor2.core_solver import CoreSolver
from gempyor2.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_dense,
    build_laplacian_tridiag,
    build_predictor_corrector,
)
from gempyor2.model_core import ModelCore

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth multi-mode initial condition on [0, 1] with zero Dirichlet BCs."""
    return (
        np.sin(np.pi * x)
        + 0.3 * np.sin(3.0 * np.pi * x)
        + 0.1 * np.sin(5.0 * np.pi * x)
    )


# ---------------------------------------------------------------------
# Test 1: IMEX reduces to CN when reaction term is zero
# ---------------------------------------------------------------------


def test_imex_reduces_to_cn_when_reaction_zero() -> None:
    """IMEX run_imex with zero reaction matches pure CN diffusion.

    We run two simulations of the same pure diffusion problem with
    Neumann boundary conditions:

    1. Direct Crank-Nicolson using CoreSolver.run with rhs = state.
    2. IMEX predictor-corrector using CoreSolver.run_imex with a reaction
       RHS that is identically zero.

    Because both paths use the same discrete operator and boundary
    treatment, the results at the final time should agree up to
    numerical tolerance.
    """
    diffusivity = 0.1
    n = 32
    total_time = 0.5
    n_steps = 51
    time_grid = np.linspace(0.0, total_time, n_steps)

    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    dt = time_grid[1] - time_grid[0]

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=diffusivity, bc="neumann")

    # Initial condition
    y0 = _initial_condition(x).astype(float).reshape(n, 1)

    # --- Reference: pure CN via build_crank_nicolson_dense + run() ---

    left_cn, right_cn = build_crank_nicolson_dense(geom, cfg, dt)

    core_cn = ModelCore(
        n_states=n,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=False,
    )
    core_cn.set_initial_state(y0)

    def rhs_cn(
        t: float,  # noqa: ARG001
        state: np.ndarray,
    ) -> np.ndarray:
        # For pure diffusion with CN, rhs is just the current state.
        return state

    solver_cn = CoreSolver(core_cn, operators=(left_cn, right_cn))
    solver_cn.run(rhs_cn)

    y_cn = core_cn.current_state[:, 0].copy()

    # --- IMEX: reaction term identically zero, run_imex() ---

    lap = build_laplacian_tridiag(
        n=geom.n,
        dx=geom.dx,
        coeff=cfg.coeff,
        dtype=np.float64,
        bc=cfg.bc,
    ).toarray()
    time_scaled_op = dt * lap
    predictor, left_imex, right_imex = build_predictor_corrector(time_scaled_op)

    core_imex = ModelCore(
        n_states=n,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=False,
    )
    core_imex.set_initial_state(y0)

    def reaction_zero(
        t: float,  # noqa: ARG001
        state: np.ndarray,
    ) -> np.ndarray:
        # No reaction: F(t, y) = 0
        return np.zeros_like(state)

    solver_imex = CoreSolver(core_imex, operators=(predictor, left_imex, right_imex))
    solver_imex.run_imex(reaction_zero)

    y_imex = core_imex.current_state[:, 0].copy()

    # They should agree closely (up to numerical differences in how operators were built)
    assert np.allclose(y_imex, y_cn, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------
# Test 2: IMEX reaction-diffusion temporal convergence (linear reaction)
# ---------------------------------------------------------------------


def test_imex_reaction_diffusion_temporal_convergence() -> None:
    """IMEX predictor-corrector shows temporal convergence for linear rxn-diffusion.

    We consider the semi-discrete system

        y' = A y + λ y,

    where A is the discrete diffusion operator (D * Δ_h) with absorbing
    boundary conditions and λ is a scalar reaction rate. This is a linear
    ODE system, so the exact solution is

        y(T) = exp((A + λ I) * T) @ y0.

    We integrate this system using CoreSolver.run_imex with:

        - implicit CN on A via (predictor, L_op, R_op),
        - explicit Heun predictor-corrector on the reaction term λ y.

    We then verify that the error at final time decreases with dt and
    exhibits at least first-order convergence (we expect ~second order).
    """
    diffusivity = 0.1
    lambda_react = -0.3
    total_time = 0.2

    n = 32
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=diffusivity, bc="absorbing")

    # Build spatial operator A = D * Δ_h (discrete diffusion)
    lap = build_laplacian_tridiag(
        n=geom.n,
        dx=geom.dx,
        coeff=cfg.coeff,
        dtype=np.float64,
        bc=cfg.bc,
    ).toarray()

    # Full linear operator B = A + λ I for the exact solution
    identity_mat = np.eye(n, dtype=float)
    operator_full = lap + lambda_react * identity_mat

    # Initial condition and exact solution
    y0 = _initial_condition(x).astype(float)
    y_exact = expm(total_time * operator_full) @ y0

    # Time step refinements: dt halves approximately between runs
    n_steps_list = [41, 81, 161]  # dt ≈ total_time / (n_steps - 1)

    errors: list[float] = []
    dts: list[float] = []

    for n_steps in n_steps_list:
        time_grid = np.linspace(0.0, total_time, n_steps)
        dt = time_grid[1] - time_grid[0]

        # IMEX operators: predictor (identity), and CN operators for A
        time_scaled_op = dt * lap
        predictor, left_op, right_op = build_predictor_corrector(time_scaled_op)

        core = ModelCore(
            n_states=n,
            n_subgroups=1,
            time_grid=time_grid,
            store_history=False,
        )
        core.set_initial_state(y0.reshape(n, 1))

        def reaction_rhs(
            t: float,  # noqa: ARG001
            state: np.ndarray,
        ) -> np.ndarray:
            # Linear reaction term: λ y
            return lambda_react * state

        solver = CoreSolver(core, operators=(predictor, left_op, right_op))
        solver.run_imex(reaction_rhs)

        y_num = core.current_state[:, 0].copy()
        err = norm(y_num - y_exact, ord=np.inf)

        errors.append(err)
        dts.append(dt)

    errors_arr = np.array(errors)
    dts_arr = np.array(dts)

    # Errors should decrease as dt decreases
    assert errors_arr[0] > errors_arr[-1], (
        f"IMEX temporal errors not decreasing: {errors_arr}"
    )

    # Estimate convergence order from the last two refinements
    order = np.log(errors_arr[-2] / errors_arr[-1]) / np.log(
        dts_arr[-2] / dts_arr[-1],
    )

    # We expect ~2nd order; require at least > 1.0 to be robust
    assert order > 1.0, f"IMEX temporal order too low: got {order}"
