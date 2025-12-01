"""Tests for matrix operations in gempyor2.matrix_ops module."""

from dataclasses import dataclass

import numpy as np
import pytest

from gempyor2.matrix_ops import (
    DiffusionConfig,
    GridGeometry,
    build_crank_nicolson_dense,
    build_crank_nicolson_sparse,
    build_laplacian_tridiag,
    build_predictor_corrector,
    encode_dense_groups,
    grouped_count_ids,
    grouped_sum_ids,
    grouped_sum_ids_2d,
    implicit_solve,
    matrix_grouped_count,
    matrix_grouped_sum,
    smooth,
    solver_dispatcher,
)


def _as_dense(mat: np.ndarray | object) -> np.ndarray:
    """Convert matrix to dense ndarray if needed.

    Args:
        mat: Input matrix, possibly sparse.

    Returns:
        Dense ndarray version of the input matrix.
    """
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


@dataclass
class _DummyRule:
    """Simple dummy rule with ``nonlinear`` and ``stochastic`` flags."""

    nonlinear: bool = False
    stochastic: bool = False


# -------------------------------------------------------------------
# Laplacian & CN operators
# -------------------------------------------------------------------


def test_laplacian_tridiag_neumann_structure_small() -> None:
    """Test structure of Laplacian tridiagonal matrix with Neumann BCs."""
    n = 5
    dx = 1.0
    coeff = 1.0
    a = build_laplacian_tridiag(n, dx, coeff, bc="neumann")
    a_dense = a.toarray()

    # Check symmetry
    assert np.allclose(a_dense, a_dense.T)

    main_diag = np.diag(a_dense)
    off1 = np.diag(a_dense, k=1)
    offm1 = np.diag(a_dense, k=-1)

    # Endpoints -1, interior -2 (up to scale factor = coeff/dx^2 = 1)
    assert main_diag[0] == pytest.approx(-1.0)
    assert main_diag[-1] == pytest.approx(-1.0)
    assert np.allclose(main_diag[1:-1], -2.0)

    # Off-diagonals all ones
    assert np.allclose(off1, 1.0)
    assert np.allclose(offm1, 1.0)


def test_laplacian_tridiag_absorbing_structure_small() -> None:
    """Test structure of Laplacian tridiagonal matrix with absorbing BCs."""
    n = 5
    dx = 1.0
    coeff = 1.0
    a = build_laplacian_tridiag(n, dx, coeff, bc="absorbing")
    a_dense = a.toarray()

    main_diag = np.diag(a_dense)
    # Endpoints -2, interior -2
    assert main_diag[0] == pytest.approx(-2.0)
    assert main_diag[-1] == pytest.approx(-2.0)
    assert np.allclose(main_diag[1:-1], -2.0)


def test_crank_nicolson_dense_vs_sparse_equivalent() -> None:
    """Test that dense and sparse Crank-Nicolson operators are equivalent."""
    n = 10
    dx = 0.1
    coeff = 0.5
    dt = 0.01

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    left_dense, right_dense = build_crank_nicolson_dense(geom, cfg, dt)
    left_sparse, right_sparse = build_crank_nicolson_sparse(geom, cfg, dt)

    assert np.allclose(left_dense, left_sparse.toarray())
    assert np.allclose(right_dense, right_sparse.toarray())


# -------------------------------------------------------------------
# Predictor-corrector algebra
# -------------------------------------------------------------------


def test_build_predictor_corrector_consistent_dense() -> None:
    """Test predictor-corrector matrices for a dense base matrix.

    With the new semantics, predictor should be the identity and
    left/right operators should be I ± 0.5 * base_matrix.
    """
    n = 8
    dx = 0.1
    coeff = 0.3
    a = build_laplacian_tridiag(n, dx, coeff).toarray()

    predictor, lc, rc = build_predictor_corrector(a)

    ident = np.eye(n, dtype=a.dtype)
    assert np.allclose(predictor, ident)
    assert np.allclose(lc, ident - 0.5 * a)
    assert np.allclose(rc, ident + 0.5 * a)


def test_build_predictor_corrector_consistent_sparse() -> None:
    """Test predictor-corrector matrices for a sparse base matrix.

    Predictor should be identity; left/right are I ± 0.5 * base_matrix.
    """
    n = 8
    dx = 0.1
    coeff = 0.3
    a = build_laplacian_tridiag(n, dx, coeff)  # sparse

    predictor, left_corrector, right_corrector = build_predictor_corrector(a)

    a_dense = a.toarray()
    ident = np.eye(n, dtype=a_dense.dtype)

    assert np.allclose(_as_dense(predictor), ident)
    assert np.allclose(_as_dense(left_corrector), ident - 0.5 * a_dense)
    assert np.allclose(_as_dense(right_corrector), ident + 0.5 * a_dense)


def test_predictor_corrector_lr_match_crank_nicolson_dense() -> None:
    """PC left/right operators match CN when built from the same time-scaled Laplacian."""
    n = 12
    dx = 0.1
    coeff = 0.4
    dt = 0.03

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    # CN operators from the high-level builder
    cn_left, cn_right = build_crank_nicolson_dense(geom, cfg, dt)

    # PC operators from the same time-scaled operator
    lap = build_laplacian_tridiag(n, dx, coeff)
    time_scaled = lap.toarray() * dt
    predictor, pc_left, pc_right = build_predictor_corrector(time_scaled)

    ident = np.eye(n, dtype=time_scaled.dtype)
    assert np.allclose(predictor, ident)
    assert np.allclose(pc_left, cn_left)
    assert np.allclose(pc_right, cn_right)


def test_predictor_corrector_lr_match_crank_nicolson_sparse() -> None:
    """PC left/right operators match CN (sparse) with the same time-scaled Laplacian."""
    n = 12
    dx = 0.1
    coeff = 0.4
    dt = 0.03

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    cn_left, cn_right = build_crank_nicolson_sparse(geom, cfg, dt)

    lap = build_laplacian_tridiag(n, dx, coeff)  # sparse
    time_scaled = lap * dt
    predictor, pc_left, pc_right = build_predictor_corrector(time_scaled)

    ident = np.eye(n, dtype=cn_left.dtype)
    assert np.allclose(_as_dense(predictor), ident)
    assert np.allclose(pc_left.toarray(), cn_left.toarray())
    assert np.allclose(pc_right.toarray(), cn_right.toarray())


# -------------------------------------------------------------------
# implicit_solve correctness (dense & sparse)
# -------------------------------------------------------------------


def test_implicit_solve_dense_matches_direct() -> None:
    """Test implicit_solve with dense operators matches direct solve."""
    n = 10
    dx = 0.1
    coeff = 0.2
    dt = 0.05

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    # Use dense CN operators
    left, right = build_crank_nicolson_dense(geom, cfg, dt)

    x = np.linspace(0.0, 1.0, n)
    rhs = right @ x
    y_direct = np.linalg.solve(left, rhs)
    y = implicit_solve(left, right, x)

    assert np.allclose(y, y_direct, atol=1e-10, rtol=1e-10)


def test_implicit_solve_sparse_matches_dense() -> None:
    """Test implicit_solve with sparse operators matches dense version."""
    n = 20
    dx = 0.05
    coeff = 0.1
    dt = 0.02

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    left_dense, right_dense = build_crank_nicolson_dense(geom, cfg, dt)
    left_sparse, right_sparse = build_crank_nicolson_sparse(geom, cfg, dt)

    rng = np.random.default_rng(7)
    x = rng.standard_normal(size=n)

    y_dense = implicit_solve(left_dense, right_dense, x)
    y_sparse = implicit_solve(left_sparse, right_sparse, x)

    assert np.allclose(y_sparse, y_dense, atol=1e-10, rtol=1e-10)


def test_implicit_solve_2d_rhs_matches_columnwise() -> None:
    """Test implicit_solve with 2D RHS matches column-wise solves."""
    n = 8
    dx = 0.1
    coeff = 0.2
    dt = 0.03

    geom = GridGeometry(n=n, dx=dx)
    cfg = DiffusionConfig(coeff=coeff)

    left, right = build_crank_nicolson_dense(geom, cfg, dt)

    rng = np.random.default_rng(6)
    x = rng.standard_normal(size=(n, 3))

    y_col = np.column_stack([
        np.linalg.solve(left, right @ x[:, k]) for k in range(x.shape[1])
    ])
    y = implicit_solve(left, right, x)

    assert np.allclose(y, y_col, atol=1e-10, rtol=1e-10)


# -------------------------------------------------------------------
# Group ops & IDs equivalence
# -------------------------------------------------------------------


def test_grouped_sum_ids_matches_matrix_grouped_sum() -> None:
    """Test that grouped_sum_ids matches matrix_grouped_sum for 1D values."""
    rng = np.random.default_rng(0)
    n_rep = 100
    n_groups = 5

    group_ids = rng.integers(0, n_groups, size=n_rep)
    values = rng.normal(size=n_rep)
    group_matrix = encode_dense_groups(group_ids, n_groups)

    sum_matrix = matrix_grouped_sum(group_matrix, values)
    sum_ids = grouped_sum_ids(values, group_ids, n_groups)

    assert np.allclose(sum_ids, sum_matrix)


def test_grouped_sum_ids_2d_matches_matrix_grouped_sum() -> None:
    """Test that grouped_sum_ids_2d matches matrix_grouped_sum for 2D values."""
    rng = np.random.default_rng(1)
    n_rep = 50
    n_groups = 4
    k = 3

    group_ids = rng.integers(0, n_groups, size=n_rep)
    values = rng.normal(size=(n_rep, k))

    group_matrix = encode_dense_groups(group_ids, n_groups)

    sum_matrix = matrix_grouped_sum(group_matrix, values)
    sum_ids = grouped_sum_ids_2d(values, group_ids, n_groups)

    assert np.allclose(sum_ids, sum_matrix)


def test_grouped_count_ids_matches_matrix_grouped_count() -> None:
    """Test that grouped_count_ids matches matrix_grouped_count."""
    rng = np.random.default_rng(2)
    n_rep = 100
    n_groups = 7

    group_ids = rng.integers(0, n_groups, size=n_rep)
    group_matrix = encode_dense_groups(group_ids, n_groups)

    count_matrix = matrix_grouped_count(group_matrix)
    count_ids = grouped_count_ids(group_ids, n_groups)

    assert np.allclose(count_ids, count_matrix)


# -------------------------------------------------------------------
# smooth behavior
# -------------------------------------------------------------------


def test_smooth_alpha_zero_identity() -> None:
    """Test that smooth with alpha=0 returns the input unchanged."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 5, 6))

    y = smooth(x, alpha=0.0)
    assert np.allclose(y, x)


def test_smooth_alpha_one_full_mean() -> None:
    """Test that smooth with alpha=1 returns the mean along the last axis."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 5, 6))

    y = smooth(x, alpha=1.0)
    mean_last = x.mean(axis=-1, keepdims=True)

    assert np.allclose(y, mean_last)


def test_smooth_out_argument_used() -> None:
    """Test that the out argument is used and returned by smooth."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 5, 6))
    out = np.empty_like(x)

    y = smooth(x, alpha=0.5, out=out)

    assert y is out
    # Just sanity: should be between x and its mean
    mean_last = x.mean(axis=-1, keepdims=True)
    expected = 0.5 * x + 0.5 * mean_last
    assert np.allclose(y, expected)


# -------------------------------------------------------------------
# solver_dispatcher behavior (new signature with dt)
# -------------------------------------------------------------------


def test_solver_dispatcher_crank_nicolson_for_linear_rules() -> None:
    """solver_dispatcher uses Crank-Nicolson for purely linear rules."""
    rules = [_DummyRule(nonlinear=False, stochastic=False)]
    geom = GridGeometry(n=16, dx=0.1)
    cfg = DiffusionConfig(coeff=0.3)
    dt = 0.01

    method, operators = solver_dispatcher(rules, geom, cfg, dt)

    assert method == "crank-nicolson"
    assert len(operators) == 2
    left_op, right_op = operators
    # Basic shape sanity checks
    assert left_op.shape == (geom.n, geom.n)
    assert right_op.shape == (geom.n, geom.n)


def test_solver_dispatcher_predictor_corrector_for_nonlinear_or_stochastic() -> None:
    """solver_dispatcher is predictor-corrector when rules are nonlinear/stochastic."""
    rules = [
        _DummyRule(nonlinear=True, stochastic=False),
        _DummyRule(nonlinear=False, stochastic=True),
    ]
    geom = GridGeometry(n=16, dx=0.1)
    cfg = DiffusionConfig(coeff=0.3)
    dt = 0.01

    method, operators = solver_dispatcher(rules, geom, cfg, dt)

    assert method == "predictor-corrector"
    assert len(operators) == 3
    predictor, left_op, right_op = operators
    # Basic shape sanity checks
    assert predictor.shape == (geom.n, geom.n)
    assert left_op.shape == (geom.n, geom.n)
    assert right_op.shape == (geom.n, geom.n)
