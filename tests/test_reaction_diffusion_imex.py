# tests/test_reaction_diffusion_imex.py
"""Tests for IMEX predictor-corrector reaction-diffusion stepping."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.linalg import expm

from gempyor2.core_solver import CoreSolver, ReactionRHSFunction, RHSFunction
from gempyor2.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_dense,
    build_laplacian_tridiag,
    build_predictor_corrector,
)
from gempyor2.model_core import ModelCore

FloatArray = NDArray[np.floating]

# ---------------------------------------------------------------------
# Module-level constants (avoid extra locals in tests)
# ---------------------------------------------------------------------

DIFFUSIVITY = 0.1
N_POINTS = 32

TOTAL_TIME_CN = 0.5
N_STEPS_CN = 51

TOTAL_TIME_RD = 0.2
LAMBDA_REACT = -0.3
N_STEPS_LIST = (41, 81, 161)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _initial_condition(x: FloatArray) -> FloatArray:
    """
    Smooth multi-mode initial condition on [0, 1] with zero Dirichlet BCs.

    Args:
        x: 1D array of spatial points.

    Returns:
        Initial condition evaluated at x.
    """
    result = (
        np.sin(np.pi * x)
        + 0.3 * np.sin(3.0 * np.pi * x)
        + 0.1 * np.sin(5.0 * np.pi * x)
    )
    # Help mypy: ensure this is seen as an ndarray[float, ...]
    return cast("FloatArray", np.asarray(result, dtype=float))


def _run_cn_diffusion(
    y0: FloatArray,
    time_grid: FloatArray,
    geom: GridGeometry,
    cfg: DiffusionConfig,
) -> FloatArray:
    """
    Run pure diffusion using Crank-Nicolson via CoreSolver.run.

    Args:
        y0: Initial condition array of shape (n_points, 1).
        time_grid: 1D array of time points.
        geom: GridGeometry object defining spatial grid.
        cfg: DiffusionConfig object defining diffusion parameters.

    Returns:
        Final solution array of shape (n_points,).
    """
    dt = float(time_grid[1] - time_grid[0])
    left_cn, right_cn = build_crank_nicolson_dense(geom, cfg, dt)

    core = ModelCore(
        n_states=y0.shape[0],
        n_subgroups=1,
        time_grid=time_grid,
        store_history=False,
    )
    core.set_initial_state(y0)

    def rhs_cn(
        _t: float,
        state: FloatArray,
    ) -> FloatArray:
        """
        For pure diffusion with CN, rhs is just the current state.

        Args:
            _t: Current time (unused).
            state: Current state array.

        Returns:
            The state itself.
        """
        return state

    solver = CoreSolver(core, operators=(left_cn, right_cn))
    solver.run(cast("RHSFunction", rhs_cn))

    return core.current_state[:, 0].copy()


def _run_imex_diffusion_zero_reaction(
    y0: FloatArray,
    time_grid: FloatArray,
    geom: GridGeometry,
    cfg: DiffusionConfig,
) -> FloatArray:
    """
    Run diffusion using IMEX with a reaction term identically zero.

    Args:
        y0: Initial condition array of shape (n_points, 1).
        time_grid: 1D array of time points.
        geom: GridGeometry object defining spatial grid.
        cfg: DiffusionConfig object defining diffusion parameters.

    Returns:
        Final solution array of shape (n_points,).
    """
    dt = float(time_grid[1] - time_grid[0])

    lap = build_laplacian_tridiag(
        n=geom.n,
        dx=geom.dx,
        coeff=cfg.coeff,
        dtype=np.float64,
        bc=cfg.bc,
    ).toarray()
    time_scaled_op = dt * lap
    predictor, left_imex, right_imex = build_predictor_corrector(time_scaled_op)

    core = ModelCore(
        n_states=y0.shape[0],
        n_subgroups=1,
        time_grid=time_grid,
        store_history=False,
    )
    core.set_initial_state(y0)

    def reaction_zero(
        t: float,  # noqa: ARG001
        state: FloatArray,
    ) -> FloatArray:
        # No reaction: F(t, y) = 0
        return np.zeros_like(state)

    solver = CoreSolver(core, operators=(predictor, left_imex, right_imex))
    solver.run_imex(cast("ReactionRHSFunction", reaction_zero))

    return core.current_state[:, 0].copy()


def _build_linear_reaction_diffusion_setup() -> tuple[
    FloatArray, FloatArray, FloatArray
]:
    """
    Build spatial operator, initial condition, and exact solution.

    We consider the semi-discrete system y' = A y + λ y, where A is the
    discrete diffusion operator with absorbing BCs and λ is a scalar reaction rate.
    The exact solution at time T is y(T) = exp((A + λ I) * T) @ y0.

    Args:
        None.

    Returns:
        Tuple of (diffusion operator A, initial condition y0, exact solution y_exact).
    """
    x = np.linspace(0.0, 1.0, N_POINTS)
    geom = GridGeometry(n=N_POINTS, dx=x[1] - x[0])
    cfg = DiffusionConfig(coeff=DIFFUSIVITY, bc="absorbing")

    # Build spatial operator A = D * Δ_h (discrete diffusion)
    lap = build_laplacian_tridiag(
        n=geom.n,
        dx=geom.dx,
        coeff=cfg.coeff,
        dtype=np.float64,
        bc=cfg.bc,
    ).toarray()

    # Full linear operator B = A + λ I for the exact solution
    identity_mat = np.eye(N_POINTS, dtype=float)
    operator_full = lap + LAMBDA_REACT * identity_mat

    # Initial condition and exact solution
    y0 = _initial_condition(cast("FloatArray", x)).astype(float)
    y_exact = expm(TOTAL_TIME_RD * operator_full) @ y0

    return cast("FloatArray", lap), y0, cast("FloatArray", y_exact)


def _imex_temporal_error(
    lap: FloatArray,
    y0: FloatArray,
    y_exact: FloatArray,
    n_steps: int,
) -> tuple[float, float]:
    """
    Run IMEX for n_steps.

    Args:
        lap: Discrete diffusion operator A.
        y0: Initial condition array.
        y_exact: Exact solution array at final time.
        n_steps: Number of time steps.

    Returns:
        Tuple of (dt, error at final time).
    """
    time_grid = np.linspace(0.0, TOTAL_TIME_RD, n_steps)
    dt = float(time_grid[1] - time_grid[0])

    time_scaled_op = dt * lap
    predictor, left_op, right_op = build_predictor_corrector(time_scaled_op)

    core = ModelCore(
        n_states=N_POINTS,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=False,
    )
    core.set_initial_state(y0.reshape(N_POINTS, 1))

    def reaction_rhs(
        _t: float,
        state: FloatArray,
    ) -> FloatArray:
        """
        Linear reaction term F(t, y) = λ y.

        Args:
            _t: Current time (unused).
            state: Current state array.

        Returns:
            Reaction term array.
        """
        return LAMBDA_REACT * state

    solver = CoreSolver(core, operators=(predictor, left_op, right_op))
    solver.run_imex(cast("ReactionRHSFunction", reaction_rhs))

    y_num = core.current_state[:, 0].copy()
    err = norm(y_num - y_exact, ord=np.inf)

    # Cast to plain floats so the return type is exactly tuple[float, float]
    return float(dt), float(err)


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
    time_grid = np.linspace(0.0, TOTAL_TIME_CN, N_STEPS_CN)

    x = np.linspace(0.0, 1.0, N_POINTS, dtype=float)
    geom = GridGeometry(n=N_POINTS, dx=x[1] - x[0])
    cfg = DiffusionConfig(coeff=DIFFUSIVITY, bc="neumann")

    # Initial condition
    y0 = _initial_condition(cast("FloatArray", x)).reshape(N_POINTS, 1)

    # Reference: pure CN
    y_cn = _run_cn_diffusion(y0, time_grid, geom, cfg)

    # IMEX with zero reaction
    y_imex = _run_imex_diffusion_zero_reaction(y0, time_grid, geom, cfg)

    # They should agree closely (up to numerical diffs in how operators were built)
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
    lap, y0, y_exact = _build_linear_reaction_diffusion_setup()

    errors: list[float] = []
    dts: list[float] = []

    for n_steps in N_STEPS_LIST:
        dt, err = _imex_temporal_error(lap, y0, y_exact, n_steps)
        dts.append(dt)
        errors.append(err)

    errors_arr = np.asarray(errors, dtype=float)
    dts_arr = np.asarray(dts, dtype=float)

    # Errors should decrease as dt decreases
    assert errors_arr[0] > errors_arr[-1], (
        f"IMEX temporal errors not decreasing: {errors_arr}"
    )

    # Estimate convergence order from the last two refinements
    order = np.log(errors_arr[-2] / errors_arr[-1]) / np.log(
        dts_arr[-2] / dts_arr[-1],
    )

    # We expect ~2nd order; require at least > 1.0 to be robust
    assert float(order) > 1.0, f"IMEX temporal order too low: got {order}"
