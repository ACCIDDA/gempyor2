# tests/test_diffusion_convergence.py

"""Convergence tests for Crank-Nicolson diffusion operators."""

import numpy as np
from numpy.linalg import norm

from gempyor2.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_dense,
    implicit_solve,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _initial_condition(x: np.ndarray) -> np.ndarray:
    """Smooth multi-mode initial condition on [0, 1] with zero Dirichlet BCs."""
    return np.sin(np.pi * x) + 0.3 * np.sin(3 * np.pi * x) + 0.1 * np.sin(5 * np.pi * x)


def _analytic_solution(x: np.ndarray, diffusivity: float, t: float) -> np.ndarray:
    """Analytic diffusion solution for the multi-mode initial condition.

    This assumes the PDE

        u_t = D u_xx   on x ∈ (0, 1), t > 0

    with homogeneous Dirichlet boundary conditions u(t, 0) = u(t, 1) = 0, and
    initial condition given by _initial_condition.
    """
    return (
        np.sin(np.pi * x) * np.exp(-diffusivity * (np.pi**2) * t)
        + 0.3 * np.sin(3 * np.pi * x) * np.exp(-diffusivity * (3 * np.pi) ** 2 * t)
        + 0.1 * np.sin(5 * np.pi * x) * np.exp(-diffusivity * (5 * np.pi) ** 2 * t)
    )


# ---------------------------------------------------------------------
# Crank-Nicolson diffusion runner helpers
# ---------------------------------------------------------------------


def _run_cn_temporal(
    n_x: int,
    n_steps: int,
    diffusivity: float,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CN diffusion for temporal convergence (fix dx, refine dt).

    Uses absorbing boundary conditions, which correspond to a Dirichlet-like
    treatment in the code and match the analytic solution.
    """
    x = np.linspace(0.0, 1.0, n_x)
    dx = x[1] - x[0]
    dt = total_time / (n_steps - 1)

    geom = GridGeometry(n=n_x, dx=dx)
    cfg = DiffusionConfig(coeff=diffusivity, bc="absorbing")

    left_op, right_op = build_crank_nicolson_dense(geom, cfg, dt)

    state = _initial_condition(x)
    for _ in range(n_steps - 1):
        state = implicit_solve(left_op, right_op, state)

    return state, x


def _run_cn_spatial(
    n_x: int,
    dt: float,
    diffusivity: float,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CN diffusion for spatial convergence (fix dt extremely small)."""
    x = np.linspace(0.0, 1.0, n_x)
    dx = x[1] - x[0]

    geom = GridGeometry(n=n_x, dx=dx)
    cfg = DiffusionConfig(coeff=diffusivity, bc="absorbing")

    left_op, right_op = build_crank_nicolson_dense(geom, cfg, dt)

    state = _initial_condition(x)
    n_steps = int(total_time / dt)

    for _ in range(n_steps):
        state = implicit_solve(left_op, right_op, state)

    return state, x


# ---------------------------------------------------------------------
# CN Tests
# ---------------------------------------------------------------------


def test_diffusion_cn_temporal_convergence() -> None:
    """
    Test temporal convergence of CN diffusion with very fine spatial grid.

    Spatial error is suppressed by choosing n_x very large.
    """
    diffusivity = 0.1
    total_time = 0.25
    n_x = 1024  # spatial error negligible

    n_steps_list = [51, 101, 201]  # dt halves each refinement

    errors: list[float] = []
    dts: list[float] = []

    for n_steps in n_steps_list:
        sol, x = _run_cn_temporal(n_x, n_steps, diffusivity, total_time)
        analytic = _analytic_solution(x, diffusivity, total_time)

        err = norm(sol - analytic, ord=np.inf)
        errors.append(err)
        dts.append(total_time / (n_steps - 1))

    errors_arr = np.array(errors)
    dts_arr = np.array(dts)

    # Must be decreasing
    assert errors_arr[0] > errors_arr[-1], (
        f"Temporal errors not decreasing: {errors_arr}"
    )

    # Estimate convergence order (should be ~2, require >1)
    order = np.log(errors_arr[-2] / errors_arr[-1]) / np.log(dts_arr[-2] / dts_arr[-1])
    assert order > 1.0, f"Temporal order too low: got {order}"


def test_diffusion_cn_spatial_convergence() -> None:
    """
    Test spatial convergence of CN diffusion with very small dt.

    Temporal error negligible; dx refinement should reduce error.
    """
    diffusivity = 0.1
    total_time = 0.25
    dt = 1e-4  # tiny dt removes temporal error

    n_x_list = [64, 128, 256]  # dx halves each time

    errors: list[float] = []
    dxs: list[float] = []

    for n_x in n_x_list:
        sol, x = _run_cn_spatial(n_x, dt, diffusivity, total_time)
        analytic = _analytic_solution(x, diffusivity, total_time)

        dx = x[1] - x[0]
        dxs.append(dx)

        err = norm(sol - analytic, ord=np.inf)
        errors.append(err)

    errors_arr = np.array(errors)
    dxs_arr = np.array(dxs)

    # Must be decreasing
    assert errors_arr[0] > errors_arr[-1], (
        f"Spatial errors not decreasing: {errors_arr}"
    )

    # Spatial CN is second-order → require >1
    order = np.log(errors_arr[-2] / errors_arr[-1]) / np.log(dxs_arr[-2] / dxs_arr[-1])
    assert order > 1.0, f"Spatial order too low: got {order}"
