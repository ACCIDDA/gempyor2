# gempyor2/src/gempyor2/core_solver.py
"""Core semi-implicit solver for time-evolving models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .matrix_ops import implicit_solve

if TYPE_CHECKING:
    from .model_core import ModelCore


OPERATORS_ERROR_MSG = "operators must be a 2-tuple (CN) or 3-tuple (PC)"


# Type aliases / protocols -------------------------------------------------


class RHSFunction(Protocol):
    """Protocol for RHS functions used by CoreSolver."""

    def __call__(
        self,
        t: float,
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute the RHS array for the solver step.

        Args:
            t: Current simulation time.
            state: Current state array of shape (n_states, n_subgroups).

        Returns:
            NDArray[np.floating]: RHS array of shape
            (n_states, n_subgroups) to be fed into the solver step.
        """


CoreOperators2 = tuple[NDArray[np.floating], NDArray[np.floating]]
CoreOperators3 = tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
CoreOperators = CoreOperators2 | CoreOperators3


class CoreSolver:
    """Semi-implicit hybrid solver operating on a ModelCore time/state grid.

    This solver:

    - Uses a time- and state-aware RHS function to apply rule-driven dynamics.
    - Supports Crank-Nicolson or predictor-corrector integration via
      pre-built operators.
    - Falls back to a direct assignment "Euler" mode when no operators are
      given.

    Notes:
        This preserves the semantics of the original EpiSolver:

        - The `rhs` argument to `step` is interpreted as the quantity that is
          directly fed into the implicit solve (typically a provisional next
          state or regularized quantity), *not* necessarily dX/dt.
        - In the no-operator case, `next_state = rhs` (no dt scaling).
    """

    def __init__(
        self,
        core: ModelCore,
        operators: CoreOperators | None = None,
    ) -> None:
        """Initialize the CoreSolver.

        Args:
            core: The model core object managing state and time.
            operators: Core operators specifying the integration method.
                - For Crank-Nicolson: ``(L_op, R_op)``
                - For predictor-corrector: ``(predictor, L_op, R_op)``
                - If None, the fallback is direct assignment (Euler-like).

        Raises:
            ValueError: If ``operators`` is not None and does not have
                length 2 or 3.
        """
        self.core = core
        self.n_states = core.n_states
        self.n_subgroups = core.n_subgroups
        self.dtype = core.dtype

        # Operator configuration (matching existing EpiSolver behavior)
        if operators is None:
            self.predictor: NDArray[np.floating] | None = None
            self.L_op: NDArray[np.floating] | None = None
            self.R_op: NDArray[np.floating] | None = None
            self.method = "euler"
        elif len(operators) == 2:
            self.predictor = None
            self.L_op, self.R_op = operators
            self.method = "crank-nicolson"
        elif len(operators) == 3:
            self.predictor, self.L_op, self.R_op = operators
            self.method = "predictor-corrector"
        else:
            msg = OPERATORS_ERROR_MSG
            raise ValueError(msg)

        # Preallocate buffers to avoid per-step allocations
        self._rhs_buffer: NDArray[np.floating] = np.zeros(
            (self.n_states, self.n_subgroups),
            dtype=self.dtype,
        )
        self._next_state_buffer: NDArray[np.floating] = np.zeros_like(
            self._rhs_buffer,
        )

    # ------------------------------------------------------------------
    # Single-step advance
    # ------------------------------------------------------------------

    def step(self, rhs: NDArray[np.floating]) -> None:
        """Advance one timestep using the configured implicit stepping.

        Args:
            rhs: Array of shape ``(n_states, n_subgroups)``.

                Semantics:
                    - For Crank-Nicolson / predictor-corrector:
                      ``rhs`` is the quantity to which the linear
                      diffusion/regularization operators are applied (fed into
                      ``implicit_solve``).
                    - For Euler-like fallback (no operators):
                      ``rhs`` is interpreted as the next-state value directly
                      (i.e., ``next_state = rhs``).

        Raises:
            ValueError: If ``rhs`` does not have shape
                ``(n_states, n_subgroups)``.
        """
        rhs_arr = np.asarray(rhs, dtype=self.dtype)
        if rhs_arr.shape != (self.n_states, self.n_subgroups):
            msg = (
                f"rhs shape {rhs_arr.shape} does not match expected "
                f"{(self.n_states, self.n_subgroups)}"
            )
            raise ValueError(msg)

        # Work in-place in the rhs buffer to minimize allocations
        np.copyto(self._rhs_buffer, rhs_arr)

        # Optional predictor (e.g. for predictor-corrector)
        if self.predictor is not None:
            # Matrix @ array over states/subgroups
            # (Keeps original semantics: rhs = predictor @ rhs)
            self._rhs_buffer[:] = self.predictor @ self._rhs_buffer

        # Implicit solve path (CN or PC)
        if self.L_op is not None and self.R_op is not None:
            # Apply implicit solve separately per subgroup
            for group_idx in range(self.n_subgroups):
                self._next_state_buffer[:, group_idx] = implicit_solve(
                    self.L_op,
                    self.R_op,
                    self._rhs_buffer[:, group_idx],
                )
        else:
            # Fallback: direct assignment (original "Euler" behavior)
            self._next_state_buffer[:] = self._rhs_buffer

        # Commit next state into the core
        self.core.advance_timestep(self._next_state_buffer)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self, rhs_func: RHSFunction) -> None:
        """Run the simulation over the entire time grid.

        Args:
            rhs_func: Function of the form ``rhs_func(t, state) -> rhs``,
                where:
                - ``t`` is the current simulation time.
                - ``state`` is the current state from the core, with shape
                  ``(n_states, n_subgroups)``.
                - ``rhs`` is the quantity to feed into :meth:`step`, with
                  shape ``(n_states, n_subgroups)``.

        Notes:
            - This iterates over all time points except the last (since the
              last point is the end of the final step).
            - The semantics of ``rhs`` are as in :meth:`step`: it is not
              necessarily a time derivative.
        """
        time_grid = self.core.time_grid

        # Iterate over all but the last time point
        for t in time_grid[:-1]:
            state = self.core.get_current_state()
            rhs = rhs_func(float(t), state)
            self.step(rhs)
