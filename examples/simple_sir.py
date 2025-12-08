# gempyor2/examples/simple_sir.py
"""Single-location SIR using IMEX predictor-corrector with A = 0.

This example uses the full reaction-diffusion machinery but with zero diffusion.
The linear operator A = 0, so predictor = I and the scheme becomes a pure
second-order Heun predictor-corrector method for the reaction terms in a
normalized (total population = 1.0) SIR model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gempyor2.core_solver import CoreSolver
from gempyor2.matrix_ops import build_predictor_corrector
from gempyor2.model_core import ModelCore

STATE_ARRAY_NONE_ERROR = "state_array is None despite store_history=True"


def sir_reaction(
    t: float,  # noqa: ARG001 (no explicit time dependence)
    state: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Pure reaction RHS F(t, y) for a normalized SIR model.

    The state is assumed to be normalized so that S + I + R ≈ 1.

    Args:
        t: Current time (unused; included for API compatibility).
        state: Current state array of shape (3, 1) with rows (S, I, R).
        beta: Transmission rate.
        gamma: Recovery rate.

    Returns:
        np.ndarray: Reaction RHS array F(t, state) with the same shape as
        ``state``, representing (dS/dt, dI/dt, dR/dt).
    """
    susceptibles = state[0, 0]
    infecteds = state[1, 0]

    new_inf = beta * susceptibles * infecteds
    recov = gamma * infecteds

    out = np.empty_like(state)
    out[0, 0] = -new_inf
    out[1, 0] = new_inf - recov
    out[2, 0] = recov
    return out


def run_sir_imex(
    beta: float = 0.3,
    gamma: float = 1 / 7,
    initial_infected: float = 0.01,
    total_time: float = 160.0,
    n_steps: int = 801,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a normalized SIR model using IMEX predictor-corrector with A = 0.

    The total population is normalized to 1.0, so the initial conditions are:
    S(0) = 1 - initial_infected, I(0) = initial_infected, R(0) = 0.

    Args:
        beta: Transmission rate.
        gamma: Recovery rate.
        initial_infected: Initial fraction infected (0 < I0 < 1).
        total_time: Final simulation time.
        n_steps: Number of time points in the uniform time grid.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - time_grid: 1D array of shape (n_steps,) with simulation times.
            - state_array: 3D array of shape
              (n_steps, 3, 1) containing (S, I, R) over time.

    Raises:
        RuntimeError: If the core's state_array is None despite
            store_history=True.
    """
    time_grid = np.linspace(0.0, total_time, n_steps)

    # 3 states (S, I, R), 1 subgroup
    core = ModelCore(
        n_states=3,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=True,
    )

    initial_susceptible = 1.0 - initial_infected
    initial_recovered = 0.0
    init = np.array(
        [[initial_susceptible], [initial_infected], [initial_recovered]],
        dtype=float,
    )
    core.set_initial_state(init)

    # Zero diffusion operator → base_matrix = 0
    base_matrix = np.zeros((3, 3), dtype=float)
    predictor, left_op, right_op = build_predictor_corrector(base_matrix)

    solver = CoreSolver(core, operators=(predictor, left_op, right_op))

    def reaction_f(t: float, state: np.ndarray) -> np.ndarray:
        return sir_reaction(t, state, beta=beta, gamma=gamma)

    solver.run_imex(reaction_f)

    # store_history=True → state_array is guaranteed non-None
    if core.state_array is None:
        raise RuntimeError(STATE_ARRAY_NONE_ERROR)
    return time_grid, core.state_array


def plot_sir(time: np.ndarray, states: np.ndarray) -> None:
    """Plot S, I, R trajectories from a SIR simulation.

    Args:
        time: 1D array of simulation times of shape (n_steps,).
        states: 3D array of shape (n_steps, 3, 1) containing the
            (S, I, R) trajectories returned by :func:`run_sir_imex`.
    """
    susceptibles = states[:, 0, 0]
    infecteds = states[:, 1, 0]
    recovered = states[:, 2, 0]

    plt.figure(figsize=(8, 5))
    plt.plot(time, susceptibles, label="S")
    plt.plot(time, infecteds, label="I")
    plt.plot(time, recovered, label="R")
    plt.grid(visible=True)
    plt.legend()
    plt.title("SIR via IMEX Predictor-Corrector (A = 0)")
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run and plot a simple SIR model example."""
    time, states = run_sir_imex()
    plot_sir(time, states)


if __name__ == "__main__":
    main()
