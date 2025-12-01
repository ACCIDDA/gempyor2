"""Core numerical state management for epidemic modeling.

This module provides the ModelCore class for efficient simulation of epidemic
models using preallocated multidimensional NumPy arrays.
"""

import numpy as np


class ModelCore:
    """
    Core class for managing the numerical state of an epidemic model.

    Uses preallocated multidimensional arrays for fast simulation.

    Attributes:
        state_array: ndarray of shape (n_states, n_subgroups, n_timesteps)
        time_grid: array of simulation times
        dt: timestep size
        current_step: current timestep index
    """

    def __init__(self, n_states: int, n_subgroups: int, time_grid: np.ndarray) -> None:
        """Initialize the ModelCore with simulation parameters.

        Args:
            n_states: Number of epidemic states (e.g., S, I, R)
            n_subgroups: Number of population subgroups
            time_grid: Array of simulation time points
        """
        self.time_grid = np.array(time_grid)
        self.n_states = n_states
        self.n_subgroups = n_subgroups
        self.n_timesteps = len(time_grid)
        self.dt = np.diff(time_grid).mean()  # assume uniform for now
        self.current_step = 0

        # Preallocate state array: shape (n_states, n_subgroups, n_timesteps)
        self.state_array = np.zeros(
            (self.n_states, self.n_subgroups, self.n_timesteps), dtype=np.float32
        )

    def set_initial_state(self, initial_state: np.ndarray) -> None:
        assert initial_state.shape == (self.n_states, self.n_subgroups), (
            f"Initial state shape {initial_state.shape} does not match expected {(self.n_states, self.n_subgroups)}"
        )
        self.state_array[:, :, 0] = initial_state

    def get_current_state(self) -> np.ndarray:
        return self.state_array[:, :, self.current_step]

    def get_state_at(self, step: int) -> np.ndarray:
        assert 0 <= step < self.n_timesteps, "Step out of bounds"
        return self.state_array[:, :, step]

    def apply_deltas(self, deltas: np.ndarray) -> None:
        assert deltas.shape == (self.n_states, self.n_subgroups), (
            f"Deltas shape {deltas.shape} does not match expected {(self.n_states, self.n_subgroups)}"
        )
        next_idx = self.current_step + 1
        assert next_idx < self.n_timesteps, (
            "Simulation has already reached final timestep"
        )
        self.state_array[:, :, next_idx] = (
            self.state_array[:, :, self.current_step] + deltas
        )
        self.current_step = next_idx

    def apply_next_state(self, next_state: np.ndarray) -> None:
        assert next_state.shape == (self.n_states, self.n_subgroups), (
            f"Next state shape {next_state.shape} does not match expected {(self.n_states, self.n_subgroups)}"
        )
        next_idx = self.current_step + 1
        assert next_idx < self.n_timesteps, (
            "Simulation has already reached final timestep"
        )
        self.state_array[:, :, next_idx] = next_state
        self.current_step = next_idx
