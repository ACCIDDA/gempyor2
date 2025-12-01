"""Tests for matrix operations in gempyor2.matrix_ops module."""

import numpy as np
import pytest

from gempyor2.matrix_ops import (
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
)


def _as_dense(mat: np.ndarray | object) -> np.ndarray:  # type: ignore[misc]
    """Convert matrix to dense ndarray if needed."""
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


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

    left_dense, right_dense = build_crank_nicolson_dense(n, dx, coeff)
    left_sparse, right_sparse = build_crank_nicolson_sparse(n, dx, coeff)

    assert np.allclose(left_dense, left_sparse.toarray())
    assert np.allclose(right_dense, right_sparse.toarray())


# -------------------------------------------------------------------
# Predictor-corrector algebra
# -------------------------------------------------------------------


def test_build_predictor_corrector_consistent_dense() -> None:
    """Test predictor-corrector matrices for a dense base matrix."""
    n = 8
    dx = 0.1
    coeff = 0.3
    a = build_laplacian_tridiag(n, dx, coeff).toarray()

    predictor, lc, rc = build_predictor_corrector(a)

    ident = np.eye(n, dtype=a.dtype)
    assert np.allclose(predictor, ident + a)
    assert np.allclose(lc, ident - 0.5 * a)
    assert np.allclose(rc, ident + 0.5 * a)


def test_build_predictor_corrector_consistent_sparse() -> None:
    """Test predictor-corrector matrices for a sparse base matrix."""
    n = 8
    dx = 0.1
    coeff = 0.3
    a = build_laplacian_tridiag(n, dx, coeff)  # sparse

    predictor, left_corrector, right_corrector = build_predictor_corrector(a)

    a_dense = a.toarray()
    ident = np.eye(n, dtype=a_dense.dtype)

    assert np.allclose(_as_dense(predictor), ident + a_dense)
    assert np.allclose(_as_dense(left_corrector), ident - 0.5 * a_dense)
    assert np.allclose(_as_dense(right_corrector), ident + 0.5 * a_dense)


# -------------------------------------------------------------------
# implicit_solve correctness (dense & sparse)
# -------------------------------------------------------------------


def test_implicit_solve_dense_matches_direct() -> None:
    """Test implicit_solve with dense operators matches direct solve."""
    n = 10
    dx = 0.1
    coeff = 0.2

    # Use dense CN operators
    left, right = build_crank_nicolson_dense(n, dx, coeff)

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

    left_dense, right_dense = build_crank_nicolson_dense(n, dx, coeff)
    left_sparse, right_sparse = build_crank_nicolson_sparse(n, dx, coeff)

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

    left, right = build_crank_nicolson_dense(n, dx, coeff)

    rng = np.random.default_rng(6)
    x = rng.standard_normal(size=(n, 3))

    y_col = np.column_stack(
        [np.linalg.solve(left, right @ x[:, k]) for k in range(x.shape[1])]
    )
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
