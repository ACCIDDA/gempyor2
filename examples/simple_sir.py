# gempyor2/examples/simple_sir.py
"""Single-location SIR using IMEX predictor-corrector with A = 0.

This example uses the full reaction-diffusion machinery but with zero diffusion.
The linear operator A = 0, so predictor = I and the scheme becomes a pure
second-order Heun predictor-corrector method for the reaction terms.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gempyor2.core_solver import CoreSolver
from gempyor2.matrix_ops import build_predictor_corrector
from gempyor2.model_core import ModelCore


def sir_reaction(
    t: float,  # noqa: ARG001 (no explicit time dependence)
    state: np.ndarray,
    beta: float,
    gamma: float,
    total_pop: float,
) -> np.ndarray:
    """Pure reaction RHS F(t, y) for SIR."""
    susceptibles = state[0, 0]
    infecteds = state[1, 0]

    new_inf = beta * susceptibles * infecteds / total_pop
    recov = gamma * infecteds

    out = np.empty_like(state)
    out[0, 0] = -new_inf
    out[1, 0] = new_inf - recov
    out[2, 0] = recov
    return out


def run_sir_imex(
    beta: float = 0.3,
    gamma: float = 1 / 7,
    total_pop: float = 1.0,
    initial_infected: float = 0.01,
    total_time: float = 160,
    n_steps: int = 801,
):
    """Run SIR using IMEX predictor-corrector with zero diffusion."""
    time_grid = np.linspace(0, total_time, n_steps)
    dt = time_grid[1] - time_grid[0]

    # 3 states, 1 subgroup
    core = ModelCore(
        n_states=3,
        n_subgroups=1,
        time_grid=time_grid,
        store_history=True,
    )

    initial_susceptible = total_pop - initial_infected
    initial_recovered = 0.0
    init = np.array([[initial_susceptible], [initial_infected], [initial_recovered]])
    core.set_initial_state(init)

    # Zero diffusion operator → base_matrix = 0
    base_matrix = np.zeros((3, 3), dtype=float)
    predictor, left_op, right_op = build_predictor_corrector(base_matrix)

    solver = CoreSolver(core, operators=(predictor, left_op, right_op))

    def reaction_f(t: float, state: np.ndarray):
        return sir_reaction(t, state, beta, gamma, total_pop)

    solver.run_imex(reaction_f)
    return time_grid, core.state_array


def plot_sir(time, states):
    susceptibles = states[:, 0, 0]
    infecteds = states[:, 1, 0]
    recovered = states[:, 2, 0]

    plt.figure(figsize=(8, 5))
    plt.plot(time, susceptibles, label="S")
    plt.plot(time, infecteds, label="I")
    plt.plot(time, recovered, label="R")
    plt.grid(True)
    plt.legend()
    plt.title("SIR via IMEX Predictor-Corrector (A = 0)")
    plt.show()


def main() -> None:
    """Run and plot a simple SIR model example."""
    time, states = run_sir_imex()
    plot_sir(time, states)


if __name__ == "__main__":
    main()
