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
    """Protocol for RHS functions used by CoreSolver.run."""

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
            RHS array of shape (n_states, n_subgroups) to be fed into
            the solver step.
        """


class ReactionRHSFunction(Protocol):
    """Protocol for reaction-only RHS functions used by run_imex.

    This represents the nonlinear / stochastic "reaction" part F(t, y) of
    a system

        y' = A y + F(t, y),

    where the linear part A is handled implicitly via pre-built operators
    (e.g., Crank-Nicolson) and F is treated explicitly.
    """

    def __call__(
        self,
        t: float,
        state: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute the reaction-only RHS F(t, state).

        Args:
            t: Current simulation time.
            state: Current state array of shape (n_states, n_subgroups).

        Returns:
            Reaction RHS array F(t, state) with shape
            (n_states, n_subgroups). This must not include the linear
            diffusion/coupling term A y.
        """


CoreOperators2 = tuple[NDArray[np.floating], NDArray[np.floating]]
CoreOperators3 = tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
CoreOperators = CoreOperators2 | CoreOperators3


class CoreSolver:
    """Semi-implicit hybrid solver operating on a ModelCore time/state grid.

    This solver:

    - Uses a time- and state-aware RHS function to apply rule-driven dynamics.
    - Supports Crank-Nicolson or predictor-corrector style integration via
      pre-built linear operators.
    - Falls back to a direct assignment "Euler" mode when no operators are
      given.

    There are two main usage modes:

    1. run:
        Backwards-compatible stepping where the RHS function directly supplies
        the quantity fed into the implicit solve (typically a provisional next
        state or regularized state).

    2. run_imex:
        IMEX (implicit-explicit) predictor-corrector stepping for systems of
        the form y' = A y + F(t, y), where A is linear (handled implicitly via
        Crank-Nicolson-style operators) and F is a nonlinear or stochastic
        reaction term handled explicitly.

    Notes:
        Semantics of the operators:

        - In Crank-Nicolson mode, operators = (L_op, R_op) and a single
          implicit step is

              next_state = implicit_solve(L_op, R_op, rhs),

          where rhs is supplied by the caller (typically the current state,
          possibly already regularized or modified).

        - In predictor-corrector style mode, operators = (predictor, L_op, R_op).
          Here predictor is a linear preprocessing matrix applied to rhs before
          the implicit solve:

              rhs_tilde  = predictor @ rhs
              next_state = implicit_solve(L_op, R_op, rhs_tilde).

          In the default construction via
          gempyor2.matrix_ops.build_predictor_corrector, predictor is the
          identity matrix and (L_op, R_op) encode a Crank-Nicolson treatment
          of the linear operator. Explicit predictor/corrector logic for
          nonlinear reaction terms lives in run_imex and the user-provided
          reaction RHS.

        - In the no-operator case, next_state = rhs (no dt scaling), which
          matches the original "Euler-like" behavior where the RHS directly
          provides the next state.
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

                - Crank-Nicolson:
                    (L_op, R_op)
                - Predictor-corrector style:
                    (predictor, L_op, R_op) where predictor is a
                    linear preprocessing matrix (often the identity) and
                    L_op, R_op are typically Crank-Nicolson operators
                    built from a time-scaled linear operator.
                - If None, the fallback is direct assignment (Euler-like).

        Raises:
            ValueError: If operators is not None and does not have
                length 2 or 3.
        """
        self.core = core
        self.n_states = core.n_states
        self.n_subgroups = core.n_subgroups
        self.dtype = core.dtype

        # Operator configuration
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
            rhs: Array of shape (n_states, n_subgroups).

                Semantics:
                    - For Crank-Nicolson / predictor-corrector:
                      rhs is the quantity to which the linear
                      diffusion/regularization operators are applied (fed into
                      implicit_solve), possibly after a linear preprocessing
                      step via predictor if present.
                    - For Euler-like fallback (no operators):
                      rhs is interpreted as the next-state value directly
                      (i.e., next_state = rhs).

        Raises:
            ValueError: If rhs does not have shape
                (n_states, n_subgroups).
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

        # Optional linear preprocessing (predictor) before implicit solve.
        # In the default build_predictor_corrector construction, predictor is
        # simply the identity, so this is a no-op.
        if self.predictor is not None:
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
            # Fallback: direct assignment (Euler-like behavior)
            self._next_state_buffer[:] = self._rhs_buffer

        # Commit next state into the core
        self.core.advance_timestep(self._next_state_buffer)

    # ------------------------------------------------------------------
    # Full run: legacy / direct semantics
    # ------------------------------------------------------------------

    def run(self, rhs_func: RHSFunction) -> None:
        """Run the simulation over the entire time grid.

        This method preserves the legacy semantics: the RHS function directly
        supplies the quantity that is fed into step, which then applies any
        configured linear operators.

        Args:
            rhs_func: Function of the form rhs_func(t, state) -> rhs,
                where:
                - t is the current simulation time.
                - state is the current state from the core, with shape
                  (n_states, n_subgroups).
                - rhs is the quantity to feed into step, with shape
                  (n_states, n_subgroups).

        Notes:
            - This iterates over all time points except the last (since the
              last point is the end of the final step).
            - The semantics of rhs are as in step: it is not necessarily a
              time derivative; it can be a provisional next state or a
              regularized quantity that is then passed through the linear
              operators configured for the solver.
        """
        time_grid = self.core.time_grid

        # Iterate over all but the last time point
        for t in time_grid[:-1]:
            state = self.core.get_current_state()
            rhs = rhs_func(float(t), state)
            self.step(rhs)

    # ------------------------------------------------------------------
    # Full run: IMEX predictor-corrector for reaction-diffusion
    # ------------------------------------------------------------------

    def run_imex(self, rhs_func: ReactionRHSFunction) -> None:
        """Run an IMEX predictor-corrector scheme over the time grid.

        This method is intended for systems of the form

            y' = A y + F(t, y),

        where:

        - The linear operator A (e.g., diffusion / coupling) is encoded in
          the pre-built operators (L_op, R_op), typically via a
          Crank-Nicolson construction on dt * A.
        - The nonlinear / stochastic reaction term F(t, y) is provided by
          rhs_func and handled explicitly in time.

        The time-stepping scheme per interval [t_n, t_{n+1}] is:

            1. f_n    = F(t_n, y_n)
            2. y_pred = y_n + dt * f_n          (explicit Euler predictor)
            3. f_pred = F(t_{n+1}, y_pred)
            4. x      = y_n + 0.5 * dt * (f_n + f_pred)
            5. y_{n+1} = implicit_solve(L_op, R_op, x)

        where step 5 is implemented by passing x into step, which applies any
        configured linear preprocessing (predictor) followed by the implicit
        solve.

        Args:
            rhs_func: Reaction-only RHS function F(t, state) that computes
                the nonlinear / stochastic part of the dynamics without the
                linear operator A. It must return an array of shape
                (n_states, n_subgroups).

        Raises:
            RuntimeError: If no linear operators are configured (i.e., method
                is "euler" and L_op/R_op are None).
        """
        if self.L_op is None or self.R_op is None:
            msg = (
                "run_imex requires configured linear operators (L_op, R_op); "
                "got method without implicit operators."
            )
            raise RuntimeError(msg)

        time_grid = self.core.time_grid
        dt = float(self.core.dt)

        # Iterate over all but the last time point
        for idx in range(len(time_grid) - 1):
            t_n = float(time_grid[idx])
            t_np1 = float(time_grid[idx + 1])

            state_n = self.core.get_current_state()

            # Explicit reaction predictor-corrector (Heun/trapezoidal)
            f_n = rhs_func(t_n, state_n)
            state_pred = state_n + dt * f_n
            f_pred = rhs_func(t_np1, state_pred)

            rhs_linear = state_n + 0.5 * dt * (f_n + f_pred)

            # Apply implicit solve for the linear part via step()
            self.step(rhs_linear)
