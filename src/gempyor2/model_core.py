# gempyor2/src/gempyor2/model_core.py
"""Core class for managing the numerical state of a model."""

from __future__ import annotations

import numpy as np

# Error / message constants -------------------------------------------------

TIMEGRID_1D_ERROR = "time_grid must be a 1D array"
TIMEGRID_MIN_POINTS_ERROR = "time_grid must contain at least one time point"
TIMEGRID_MONOTONE_ERROR = "time_grid must be strictly increasing"
INITIAL_STATE_SHAPE_ERROR = (
    "Initial state shape {actual} does not match expected {expected}"
)
DELTAS_SHAPE_ERROR = "Deltas shape {actual} does not match expected {expected}"
NEXT_STATE_SHAPE_ERROR = "Next state shape {actual} does not match expected {expected}"
HISTORY_NOT_STORED_ERROR = (
    "Full history is not stored (store_history=False); get_state_at is unavailable."
)
STEP_OOB_ERROR = "Step out of bounds"
FINAL_TIMESTEP_ERROR = "Simulation has already reached final timestep"


class ModelCore:
    """
    Core class for managing the numerical state of a model.

    Design goals
    ------------
    - CPU-friendly layout: per-timestep state is contiguous in memory.
    - Full state history by default, but can be disabled for very large runs.
    - Float64 throughout for numerical robustness (can be changed via dtype).

    Attributes:
    ----------
    time_grid : np.ndarray
        Array of simulation times (float64).
    dt : float
        Timestep size (assumed uniform for now).
    n_states : int
        Number of epidemic states (e.g., S, E, I1, I2, I3, R).
    n_subgroups : int
        Number of population subgroups (e.g., age groups).
    n_timesteps : int
        Number of time points in the simulation.
    current_step : int
        Index of the current timestep (0-based).
    current_state : np.ndarray
        State at the current timestep, shape (n_states, n_subgroups).
    state_array : np.ndarray | None
        Full state history, shape (n_timesteps, n_states, n_subgroups),
        or None if store_history=False.
    store_history : bool
        Whether the full time history is being stored.
    dtype : np.dtype
        Floating-point dtype used for all internal arrays (default float64).
    """

    def __init__(
        self,
        n_states: int,
        n_subgroups: int,
        time_grid: np.ndarray,
        *,
        store_history: bool = True,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        Initialize the ModelCore with simulation parameters.

        Parameters
        ----------
        n_states : int
            Number of epidemic states.
        n_subgroups : int
            Number of population subgroups (e.g., age groups).
        time_grid : array-like
            Monotonic array of simulation time points.
        store_history : bool, optional
            If True, store full state history in state_array.
            If False, only keep current_state.
        dtype : np.dtype, optional
            Floating-point dtype for internal arrays (default float64).
        """
        self.time_grid = np.asarray(time_grid, dtype=dtype)
        if self.time_grid.ndim != 1:
            msg = TIMEGRID_1D_ERROR
            raise ValueError(msg)

        self.n_states = int(n_states)
        self.n_subgroups = int(n_subgroups)
        self.n_timesteps = int(self.time_grid.size)
        self.store_history = bool(store_history)
        self.dtype = np.dtype(dtype)

        if self.n_timesteps < 1:
            msg = TIMEGRID_MIN_POINTS_ERROR
            raise ValueError(msg)

        if self.n_timesteps > 1:
            dt_arr = np.diff(self.time_grid)
            # Basic sanity check for monotonicity
            if np.any(dt_arr <= 0):
                msg = TIMEGRID_MONOTONE_ERROR
                raise ValueError(msg)
            self.dt = float(dt_arr.mean())
        else:
            # Degenerate single-step case
            self.dt = 0.0

        self.current_step: int = 0

        # Per-timestep working state (contiguous, float64 by default)
        self.current_state: np.ndarray = np.zeros(
            (self.n_states, self.n_subgroups),
            dtype=self.dtype,
        )

        # Optional full history: (n_timesteps, n_states, n_subgroups)
        if self.store_history:
            self.state_array: np.ndarray | None = np.zeros(
                (self.n_timesteps, self.n_states, self.n_subgroups),
                dtype=self.dtype,
            )
        else:
            self.state_array = None

    # ------------------------------------------------------------------
    # Initialization / accessors
    # ------------------------------------------------------------------

    def set_initial_state(self, initial_state: np.ndarray) -> None:
        """
        Set the state at t = time_grid[0].

        Parameters
        ----------
        initial_state : np.ndarray
            Array of shape (n_states, n_subgroups).
        """
        initial_state_arr = np.asarray(initial_state, dtype=self.dtype)
        if initial_state_arr.shape != (self.n_states, self.n_subgroups):
            msg = INITIAL_STATE_SHAPE_ERROR.format(
                actual=initial_state_arr.shape,
                expected=(self.n_states, self.n_subgroups),
            )
            raise ValueError(msg)

        np.copyto(self.current_state, initial_state_arr)

        if self.store_history and self.state_array is not None:
            # Store at step 0
            self.state_array[0, :, :] = self.current_state

        self.current_step = 0

    def get_current_state(self) -> np.ndarray:
        """Return the current state as a (n_states, n_subgroups) array."""
        return self.current_state

    def get_state_at(self, step: int) -> np.ndarray:
        """
        Return the state at a given timestep (requires store_history=True).

        Parameters
        ----------
        step : int
            Timestep index in [0, n_timesteps).

        Returns:
        -------
        np.ndarray
            State at the requested timestep, shape (n_states, n_subgroups).

        Raises:
        ------
        RuntimeError
            If store_history=False.
        """
        if not self.store_history or self.state_array is None:
            msg = HISTORY_NOT_STORED_ERROR
            raise RuntimeError(msg)

        if not (0 <= step < self.n_timesteps):
            msg = STEP_OOB_ERROR
            raise IndexError(msg)

        # Return a view
        return self.state_array[step, :, :]

    # ------------------------------------------------------------------
    # Stepping / updates
    # ------------------------------------------------------------------

    def _check_can_advance(self) -> None:
        if self.current_step >= self.n_timesteps - 1:
            msg = FINAL_TIMESTEP_ERROR
            raise RuntimeError(msg)

    def apply_deltas(self, deltas: np.ndarray) -> None:
        """
        Advance one timestep by applying additive deltas to the current state.

        Parameters
        ----------
        deltas : np.ndarray
            Array of shape (n_states, n_subgroups) representing state changes
            to apply over the next dt.

        Notes:
        -----
        This performs:

            next_state = current_state + deltas

        and then advances current_step, updating both current_state and
        (optionally) the history array.
        """
        deltas_arr = np.asarray(deltas, dtype=self.dtype)
        if deltas_arr.shape != (self.n_states, self.n_subgroups):
            msg = DELTAS_SHAPE_ERROR.format(
                actual=deltas_arr.shape,
                expected=(self.n_states, self.n_subgroups),
            )
            raise ValueError(msg)

        self._check_can_advance()

        # In-place update of current_state
        self.current_state += deltas_arr

        # Advance step and write to history if enabled
        self.current_step += 1
        if self.store_history and self.state_array is not None:
            self.state_array[self.current_step, :, :] = self.current_state

    def apply_next_state(self, next_state: np.ndarray) -> None:
        """
        Advance one timestep by directly specifying the next state.

        Parameters
        ----------
        next_state : np.ndarray
            Array of shape (n_states, n_subgroups) representing the state
            at the next timestep.
        """
        next_state_arr = np.asarray(next_state, dtype=self.dtype)
        if next_state_arr.shape != (self.n_states, self.n_subgroups):
            msg = NEXT_STATE_SHAPE_ERROR.format(
                actual=next_state_arr.shape,
                expected=(self.n_states, self.n_subgroups),
            )
            raise ValueError(msg)

        self._check_can_advance()

        # Overwrite current_state in-place
        np.copyto(self.current_state, next_state_arr)

        # Advance step and write to history if enabled
        self.current_step += 1
        if self.store_history and self.state_array is not None:
            self.state_array[self.current_step, :, :] = self.current_state

    def advance_timestep(self, next_state: np.ndarray) -> None:
        """Alias for apply_next_state, for solver-friendly naming."""
        self.apply_next_state(next_state)
