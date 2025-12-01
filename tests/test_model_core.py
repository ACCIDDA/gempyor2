# tests/test_model_core.py

import numpy as np
import pytest

from gempyor2.model_core import ModelCore


def test_model_core_timegrid_monotone_ok() -> None:
    """Test that ModelCore initializes correctly with a valid time grid."""
    time_grid = np.linspace(0.0, 10.0, 11)
    core = ModelCore(n_states=3, n_subgroups=2, time_grid=time_grid)

    assert core.n_states == 3
    assert core.n_subgroups == 2
    assert core.n_timesteps == 11
    assert np.isclose(core.dt, 1.0)


def test_model_core_timegrid_nonmonotone_raises() -> None:
    """Test that ModelCore raises an error with a non-monotonic time grid."""
    time_grid = np.array([0.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="strictly increasing"):
        ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)


def test_model_core_single_step_dt_zero() -> None:
    """Test that ModelCore handles single-step time grid correctly."""
    time_grid = np.array([0.0])
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)
    assert core.n_timesteps == 1
    assert core.dt == 0.0


def test_set_initial_state_and_history_on() -> None:
    """Test setting initial state with history enabled."""
    time_grid = np.linspace(0.0, 2.0, 3)  # 3 timesteps
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid, store_history=True)

    init = np.arange(6, dtype=float).reshape(2, 3)
    core.set_initial_state(init)

    assert core.current_step == 0
    assert np.array_equal(core.get_current_state(), init)
    assert core.state_array.shape == (3, 2, 3)
    assert np.array_equal(core.state_array[0], init)


def test_set_initial_state_and_history_off() -> None:
    """Test setting initial state with history disabled."""
    time_grid = np.linspace(0.0, 1.0, 3)
    core = ModelCore(
        n_states=2, n_subgroups=3, time_grid=time_grid, store_history=False
    )

    init = np.arange(6, dtype=float).reshape(2, 3)
    core.set_initial_state(init)

    assert core.current_step == 0
    assert np.array_equal(core.get_current_state(), init)
    assert core.state_array is None

    with pytest.raises(RuntimeError, match="store_history=False"):
        _ = core.get_state_at(0)


def test_set_initial_state_wrong_shape_raises() -> None:
    """Test that setting an initial state with the wrong shape raises an error."""
    time_grid = np.linspace(0.0, 1.0, 3)
    core = ModelCore(n_states=2, n_subgroups=3, time_grid=time_grid)

    bad_init = np.zeros((3, 2))
    with pytest.raises(ValueError, match="Initial state shape"):
        core.set_initial_state(bad_init)


def test_apply_deltas_updates_state_and_history() -> None:
    """Test that applying deltas updates the state and history correctly."""
    time_grid = np.linspace(0.0, 2.0, 3)  # steps: 0, 1, 2
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid, store_history=True)

    init = np.zeros((2, 2), dtype=float)
    core.set_initial_state(init)

    delta1 = np.ones((2, 2), dtype=float)
    delta2 = 2.0 * np.ones((2, 2), dtype=float)

    core.apply_deltas(delta1)
    assert core.current_step == 1
    assert np.array_equal(core.get_current_state(), init + delta1)
    assert np.array_equal(core.get_state_at(1), init + delta1)

    core.apply_deltas(delta2)
    assert core.current_step == 2
    expected = init + delta1 + delta2
    assert np.array_equal(core.get_current_state(), expected)
    assert np.array_equal(core.get_state_at(2), expected)


def test_apply_deltas_history_off() -> None:
    """Test applying deltas with history disabled."""
    time_grid = np.linspace(0.0, 2.0, 3)
    core = ModelCore(
        n_states=2, n_subgroups=2, time_grid=time_grid, store_history=False
    )

    init = np.zeros((2, 2), dtype=float)
    core.set_initial_state(init)

    delta = np.ones((2, 2), dtype=float)
    core.apply_deltas(delta)

    assert core.current_step == 1
    assert np.array_equal(core.get_current_state(), init + delta)
    assert core.state_array is None

    with pytest.raises(RuntimeError):
        _ = core.get_state_at(1)


def test_apply_next_state_overwrites_state() -> None:
    """Test that applying next state overwrites the current state correctly."""
    time_grid = np.linspace(0.0, 1.0, 3)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid, store_history=True)

    init = np.zeros((2, 2), dtype=float)
    core.set_initial_state(init)

    next_state = np.ones((2, 2), dtype=float)
    core.apply_next_state(next_state)

    assert core.current_step == 1
    assert np.array_equal(core.get_current_state(), next_state)
    assert np.array_equal(core.get_state_at(1), next_state)


def test_cannot_advance_past_last_step() -> None:
    """Test that applying next state at the final timestep raises an error."""
    time_grid = np.array([0.0, 1.0])
    core = ModelCore(n_states=1, n_subgroups=1, time_grid=time_grid)

    core.set_initial_state(np.array([[0.0]]))
    core.apply_next_state(np.array([[1.0]]))

    assert core.current_step == 1
    with pytest.raises(RuntimeError, match="final timestep"):
        core.apply_next_state(np.array([[2.0]]))


def test_get_current_state_is_view() -> None:
    """Test that get_current_state returns a view into the current state."""
    time_grid = np.linspace(0.0, 1.0, 3)
    core = ModelCore(n_states=2, n_subgroups=2, time_grid=time_grid)

    init = np.zeros((2, 2), dtype=float)
    core.set_initial_state(init)

    state_view = core.get_current_state()
    state_view[0, 0] = 42.0

    # current_state should see the change
    assert core.current_state[0, 0] == 42.0
