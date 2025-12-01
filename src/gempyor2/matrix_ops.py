"""Matrix operations and linear solvers for epidemic modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, csr_matrix, diags, identity, issparse
from scipy.sparse.linalg import factorized as sparse_factorized

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike, NDArray


@dataclass
class GridGeometry:
    """Geometry of a 1D spatial grid.

    Attributes:
        n: Number of grid points.
        dx: Grid spacing.
    """

    n: int
    dx: float


@dataclass
class DiffusionConfig:
    """Configuration for diffusion-like linear operators.

    Attributes:
        coeff: Diffusion coefficient (may already include dt scaling if treated
            as a regularizer).
        dtype: Floating dtype (e.g. np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".
    """

    coeff: float
    dtype: DTypeLike = np.float64
    bc: str = "neumann"


# === Internal autodispatch threshold (tuned empirically) ===
# Below this size, dense ops tend to be faster; above it, sparse is preferred.
_DISPATCH_THRESHOLD = 350

# Cache for implicit solvers (factorized L, prepped R)
# key = (id(L), id(R)) -> callable X -> Y
_IMPLICIT_SOLVER_CACHE: dict[
    tuple[int, int], Callable[[NDArray[np.floating]], NDArray[np.floating]]
] = {}

# Error message constants ---------------------------------------------------

UNKNOWN_BC_ERROR = "Unknown bc: {bc}"
VALUES_2D_ERROR = "values must be 2D (N, K)"
GROUP_IDS_LENGTH_ERROR = "group_ids and values must have the same length along axis 0"


# ======================================================================
# Core linear operators: Laplacian + Crank-Nicolson + Predictor-Corrector
# ======================================================================


def build_laplacian_tridiag(
    n: int,
    dx: float,
    coeff: float,
    dtype: DTypeLike = np.float64,
    bc: str = "neumann",
) -> csr_matrix:
    """Build a Laplacian tridiagonal matrix for a given boundary condition.

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Diffusion coefficient (may already include dt scaling if treated
            as a regularizer).
        dtype: Floating dtype (default is np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".

    Returns:
        Scaled Laplacian matrix of shape (n, n) as a CSR sparse matrix.

    Raises:
        ValueError: If bc is not "neumann" or "absorbing".
    """
    factor = coeff / dx**2
    _ = np.dtype(dtype)  # ensure valid dtype; we actually rely on default float64

    main_diag = -2.0 * np.ones(n)
    off_diag = np.ones(n - 1)

    if bc == "neumann":
        main_diag[0] = main_diag[-1] = -1.0
    elif bc == "absorbing":
        main_diag[0] = main_diag[-1] = -2.0
    else:
        msg = UNKNOWN_BC_ERROR.format(bc=bc)
        raise ValueError(msg)

    # Use Python lists so SciPy's type stubs are happy.
    laplacian = diags(
        [off_diag.tolist(), main_diag.tolist(), off_diag.tolist()],
        [-1, 0, 1],
        shape=(n, n),
    )

    scaled = laplacian * factor
    return scaled.tocsr()


def build_crank_nicolson_sparse(
    n: int,
    dx: float,
    coeff: float,
    dtype: DTypeLike = np.float64,
    bc: str = "neumann",
) -> tuple[csr_matrix, csr_matrix]:
    """Build sparse Crank-Nicolson operator matrices (L, R).

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Diffusion coefficient (may already include dt scaling if treated
            as a regularizer).
        dtype: Floating dtype (default is np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".

    Returns:
        A tuple (L, R) of CSR sparse matrices representing the left- and
        right-hand Crank-Nicolson operators.
    """
    laplacian = build_laplacian_tridiag(n, dx, coeff, dtype=dtype, bc=bc)
    identity_mat = identity(
        n,
        dtype=np.float64,
        format="csr",
    )
    left_op = identity_mat - 0.5 * laplacian
    right_op = identity_mat + 0.5 * laplacian

    if bc == "absorbing":
        for mat in (left_op, right_op):
            mat[0, :] = 0.0
            mat[0, 0] = 1.0
            mat[-1, :] = 0.0
            mat[-1, -1] = 1.0

    return cast("csr_matrix", left_op.tocsr()), cast(
        "csr_matrix",
        right_op.tocsr(),
    )


def build_crank_nicolson_dense(
    n: int,
    dx: float,
    coeff: float,
    dtype: DTypeLike = np.float64,
    bc: str = "neumann",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build dense Crank-Nicolson operator matrices (L, R).

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Diffusion coefficient (may already include dt scaling if treated
            as a regularizer).
        dtype: Floating dtype (default is np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".

    Returns:
        A tuple (L, R) of dense arrays representing the left- and
        right-hand Crank-Nicolson operators.
    """
    laplacian = build_laplacian_tridiag(n, dx, coeff, dtype=dtype, bc=bc).toarray()
    identity_mat = np.eye(n, dtype=np.float64)
    left_op = identity_mat - 0.5 * laplacian
    right_op = identity_mat + 0.5 * laplacian

    if bc == "absorbing":
        for mat in (left_op, right_op):
            mat[0, :] = 0.0
            mat[0, 0] = 1.0
            mat[-1, :] = 0.0
            mat[-1, -1] = 1.0

    return left_op, right_op


def build_crank_nicolson_operator(
    n: int,
    dx: float,
    coeff: float,
    dtype: DTypeLike = np.float64,
    bc: str = "neumann",
) -> tuple[Any, Any]:
    """Build Crank-Nicolson operators with dense/sparse autodispatch.

    Small systems are built as dense operators; large systems as sparse.

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Diffusion coefficient (may already include dt scaling if treated
            as a regularizer).
        dtype: Floating dtype (default is np.float64).
        bc: Boundary condition; either "neumann" or "absorbing".

    Returns:
        A tuple (L, R) of operator matrices, either dense arrays or CSR
        matrices depending on problem size.
    """
    if n < _DISPATCH_THRESHOLD:
        return build_crank_nicolson_dense(n, dx, coeff, dtype=dtype, bc=bc)
    return build_crank_nicolson_sparse(n, dx, coeff, dtype=dtype, bc=bc)


# --- Predictor-Corrector matrices ---


def build_predictor_corrector_dense(
    base_matrix: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Build predictor-corrector matrices for a dense base matrix.

    Args:
        base_matrix: Base operator matrix (e.g., a Laplacian) of shape
            (n, n).

    Returns:
        A tuple (predictor, L_op, R_op) of dense matrices for the
        predictor-corrector scheme.
    """
    n = base_matrix.shape[0]
    identity_mat = np.eye(n, dtype=base_matrix.dtype)
    predictor = identity_mat + base_matrix
    corrector_left = identity_mat - 0.5 * base_matrix
    corrector_right = identity_mat + 0.5 * base_matrix
    return predictor, corrector_left, corrector_right


def build_predictor_corrector_sparse(
    base_matrix: csr_matrix,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build predictor-corrector matrices for a sparse base matrix.

    Args:
        base_matrix: Base operator matrix (e.g., a Laplacian) as a CSR sparse
            matrix of shape (n, n).

    Returns:
        A tuple (predictor, L_op, R_op) of CSR sparse matrices for the
        predictor-corrector scheme.
    """
    n = base_matrix.shape[0]
    identity_mat = identity(
        n,
        format="csr",
        dtype=np.float64,
    )
    predictor = identity_mat + base_matrix
    corrector_left = identity_mat - 0.5 * base_matrix
    corrector_right = identity_mat + 0.5 * base_matrix
    return (
        cast("csr_matrix", predictor.tocsr()),
        cast("csr_matrix", corrector_left.tocsr()),
        cast("csr_matrix", corrector_right.tocsr()),
    )


def build_predictor_corrector(
    base_matrix: NDArray[np.floating] | csr_matrix,
) -> tuple[Any, Any, Any]:
    """Build predictor-corrector matrices with dense/sparse autodispatch.

    Args:
        base_matrix: Base operator matrix, either a dense array-like or a CSR
            sparse matrix.

    Returns:
        A tuple (predictor, L_op, R_op) where the matrices are dense or
        sparse depending on the input and problem size.
    """
    n = base_matrix.shape[0]
    if issparse(base_matrix) and n >= _DISPATCH_THRESHOLD:
        return build_predictor_corrector_sparse(base_matrix)

    if issparse(base_matrix):
        dense_base = np.asarray(base_matrix.toarray())
    else:
        dense_base = np.asarray(base_matrix)

    return build_predictor_corrector_dense(dense_base)


# =============================================================
# Implicit solvers: factorized (cached) & simple wrappers
# =============================================================


def _build_implicit_solver(
    left_op: NDArray[np.floating] | csr_matrix,
    right_op: NDArray[np.floating] | csr_matrix,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Build a reusable implicit solver for left_op @ y = right_op @ x.

    The returned callable solves L y = R x for y given an x.

    Args:
        left_op: Left-hand operator matrix L (dense or sparse).
        right_op: Right-hand operator matrix R (dense or sparse).

    Returns:
        A callable solver(x) that returns the solution y to
        L y = R x.
    """
    # Only treat as sparse if both operators are sparse.
    is_sparse = issparse(left_op) and issparse(right_op)

    if is_sparse:
        left_csr = cast("csr_matrix", left_op)
        right_csr = cast("csr_matrix", right_op)

        solve_left = sparse_factorized(left_csr)

        def solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
            x_arr = np.asarray(x, dtype=left_csr.dtype)
            rhs = right_csr @ x_arr
            return np.asarray(solve_left(rhs), dtype=x_arr.dtype)

    else:
        left_dense = np.asarray(left_op)
        right_dense = np.asarray(right_op)
        lu, piv = lu_factor(left_dense)

        def solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
            x_arr = np.asarray(x, dtype=left_dense.dtype)
            rhs = right_dense @ x_arr
            return np.asarray(lu_solve((lu, piv), rhs), dtype=x_arr.dtype)

    return solver


def implicit_solve(
    left_op: NDArray[np.floating] | csr_matrix,
    right_op: NDArray[np.floating] | csr_matrix,
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Perform an implicit solve with dense/sparse dispatch and caching.

    This solves left_op @ y = right_op @ x for y, reusing cached
    factorizations keyed by the operator identities.

    Args:
        left_op: Left-hand operator matrix L (dense or sparse).
        right_op: Right-hand operator matrix R (dense or sparse).
        x: Right-hand side vector or matrix.

    Returns:
        The solution array y with the same shape as x.

    Notes:
        * Uses a cached factorization keyed by (id(left_op), id(right_op)).
        * Can be safely called in tight loops (e.g., per subgroup and timestep).
    """
    key = (id(left_op), id(right_op))
    solver = _IMPLICIT_SOLVER_CACHE.get(key)
    if solver is None:
        solver = _build_implicit_solver(left_op, right_op)
        _IMPLICIT_SOLVER_CACHE[key] = solver
    return solver(np.asarray(x))


# =============================================================
# Solver meta-dispatcher
# =============================================================


def solver_dispatcher(
    rules: list[Any],
    geom: GridGeometry,
    cfg: DiffusionConfig,
) -> tuple[str, tuple[Any, ...]]:
    """Dispatch solver operator construction based on rule structure.

    Args:
        rules: List of rule objects; each must expose boolean attributes
            nonlinear and stochastic.
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration containing coefficient, dtype, and
            boundary condition.

    Returns:
        A tuple (method, operators) where:

        * method is either "crank-nicolson" or
          "predictor-corrector".
        * operators is:
            * (L_op, R_op) for Crank-Nicolson.
            * (predictor, L_op, R_op) for predictor-corrector.
    """
    n = geom.n
    dx = geom.dx
    coeff = cfg.coeff
    dtype = cfg.dtype
    bc = cfg.bc

    has_nonlinear = any(getattr(rule, "nonlinear", False) for rule in rules)
    has_stochastic = any(getattr(rule, "stochastic", False) for rule in rules)

    if has_nonlinear or has_stochastic:
        laplacian = build_laplacian_tridiag(n, dx, coeff, dtype=dtype, bc=bc)
        predictor, left_op, right_op = build_predictor_corrector(laplacian)
        operators: tuple[Any, ...] = (predictor, left_op, right_op)
        method = "predictor-corrector"
    else:
        left_op, right_op = build_crank_nicolson_operator(
            n,
            dx,
            coeff,
            dtype=dtype,
            bc=bc,
        )
        operators = (left_op, right_op)
        method = "crank-nicolson"

    return method, operators


# =============================================================
# Grouped operations: sum, count, masked sum
# =============================================================

# --- GROUPED SUM (matrix-based API) ---


def matrix_grouped_sum_sparse(
    group_matrix: csr_matrix,
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped sums using a sparse membership matrix.

    Performs (G x N) @ (N, ...) -> (G, ...).

    Args:
        group_matrix: Sparse group membership matrix of shape (G, N).
        values: Values to sum, of shape (N,) or (N, K).

    Returns:
        Grouped sums of shape (G,) or (G, K).
    """
    return np.asarray(group_matrix @ values)


def matrix_grouped_sum_dense(
    group_matrix: NDArray[np.floating],
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped sums using a dense membership matrix.

    Performs (G x N) @ (N, ...) -> (G, ...).

    Args:
        group_matrix: Dense group membership matrix of shape (G, N).
        values: Values to sum, of shape (N,) or (N, K).

    Returns:
        Grouped sums of shape (G,) or (G, K).
    """
    return np.asarray(group_matrix @ values)


def matrix_grouped_sum(
    group_matrix: csr_matrix | NDArray[np.floating],
    values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped sums with dense/sparse autodispatch.

    Args:
        group_matrix: Group membership matrix of shape (G, N), either dense
            or sparse.
        values: Values to sum, of shape (N,) or (N, K).

    Returns:
        Grouped sums of shape (G,) or (G, K).
    """
    n_groups = group_matrix.shape[0]
    if issparse(group_matrix) and n_groups >= _DISPATCH_THRESHOLD:
        return matrix_grouped_sum_sparse(group_matrix, values)
    if issparse(group_matrix):
        dense_matrix = np.asarray(group_matrix.toarray())
    else:
        dense_matrix = np.asarray(group_matrix)
    return matrix_grouped_sum_dense(dense_matrix, values)


# --- GROUPED COUNT (matrix-based API) ---


def matrix_grouped_count_sparse(
    group_matrix: csr_matrix,
) -> NDArray[np.floating]:
    """Compute grouped counts using a sparse membership matrix.

    Args:
        group_matrix: Sparse group membership matrix of shape (G, N).

    Returns:
        Group sizes as a 1D array of shape (G,).
    """
    counts = np.asarray(group_matrix.sum(axis=1)).ravel()
    return counts.astype(float)


def matrix_grouped_count_dense(
    group_matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped counts using a dense membership matrix.

    Args:
        group_matrix: Dense group membership matrix of shape (G, N).

    Returns:
        Group sizes as a 1D array of shape (G,).
    """
    counts = np.asarray(group_matrix.sum(axis=1))
    return counts.astype(float)


def matrix_grouped_count(
    group_matrix: csr_matrix | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute grouped counts with dense/sparse autodispatch.

    Args:
        group_matrix: Group membership matrix of shape (G, N), either dense
            or sparse.

    Returns:
        Group sizes as a 1D array of shape (G,).
    """
    n_groups = group_matrix.shape[0]
    if issparse(group_matrix) and n_groups >= _DISPATCH_THRESHOLD:
        return matrix_grouped_count_sparse(group_matrix)
    if issparse(group_matrix):
        dense_matrix = np.asarray(group_matrix.toarray())
    else:
        dense_matrix = np.asarray(group_matrix)
    return matrix_grouped_count_dense(dense_matrix)


# --- MASKED SUM (matrix-based API) ---


def matrix_masked_sum_sparse(
    mask_matrix: csr_matrix,
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums using a sparse mask matrix.

    Args:
        mask_matrix: Sparse mask matrix.
        data: Data to aggregate.

    Returns:
        Masked sum result as an array with shape determined by the mask and
        data.
    """
    return np.asarray(mask_matrix @ data)


def matrix_masked_sum_dense(
    mask_matrix: NDArray[np.floating],
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums using a dense mask matrix.

    Args:
        mask_matrix: Dense mask matrix.
        data: Data to aggregate.

    Returns:
        Masked sum result as an array with shape determined by the mask and
        data.
    """
    return np.asarray(mask_matrix @ data)


def matrix_masked_sum(
    mask_matrix: csr_matrix | NDArray[np.floating],
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums with dense/sparse autodispatch.

    Args:
        mask_matrix: Mask matrix, dense or sparse.
        data: Data to aggregate.

    Returns:
        Masked sum result as an array with shape determined by the mask and
        data.
    """
    n_masks = mask_matrix.shape[0]
    if issparse(mask_matrix) and n_masks >= _DISPATCH_THRESHOLD:
        return matrix_masked_sum_sparse(mask_matrix, data)
    if issparse(mask_matrix):
        dense_matrix = np.asarray(mask_matrix.toarray())
    else:
        dense_matrix = np.asarray(mask_matrix)
    return matrix_masked_sum_dense(dense_matrix, data)


# =============================================================
# Fast group-ID based paths (for age groups etc.)
# =============================================================


def grouped_count_ids(
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Compute group sizes from integer group IDs.

    Args:
        group_ids: Integer group IDs of shape (N,) in
            [0, n_groups - 1].
        n_groups: Number of groups.

    Returns:
        Counts per group as an array of shape (n_groups,).
    """
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    counts = np.bincount(group_ids_arr, minlength=n_groups)
    return counts.astype(float)


def grouped_sum_ids(
    values: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Compute grouped sums from integer group IDs for 1D values.

    Args:
        values: Values to sum of shape (N,).
        group_ids: Integer group IDs of shape (N,) in
            [0, n_groups - 1].
        n_groups: Number of groups.

    Returns:
        Grouped sums as an array of shape (n_groups,).
    """
    values_arr = np.asarray(values)
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    sums = np.bincount(group_ids_arr, weights=values_arr, minlength=n_groups)
    return sums.astype(float)


def grouped_sum_ids_2d(
    values: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Compute grouped sums from integer group IDs for 2D values.

    Args:
        values: Values to sum of shape (N, K) (e.g., compartments per
            individual).
        group_ids: Integer group IDs of shape (N,) in
            [0, n_groups - 1].
        n_groups: Number of groups.

    Returns:
        Grouped sums as an array of shape (n_groups, K).

    Raises:
        ValueError: If values is not 2D.
        ValueError: If the length of group_ids does not match the first
            dimension of values.
    """
    values_arr = np.asarray(values)
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)

    if values_arr.ndim != 2:
        msg = VALUES_2D_ERROR
        raise ValueError(msg)

    n_items, n_features = values_arr.shape
    if group_ids_arr.shape[0] != n_items:
        msg = GROUP_IDS_LENGTH_ERROR
        raise ValueError(msg)

    out = np.zeros((n_groups, n_features), dtype=values_arr.dtype)
    for feature_idx in range(n_features):
        out[:, feature_idx] = np.bincount(
            group_ids_arr,
            weights=values_arr[:, feature_idx],
            minlength=n_groups,
        )
    return out


# =============================================================
# Encoding helpers + smoothing
# =============================================================


def encode_sparse_groups(
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> csr_matrix:
    """Encode group IDs as a sparse binary membership matrix.

    Args:
        group_ids: Integer group IDs of shape (N,).
        n_groups: Number of groups.

    Returns:
        Sparse group membership matrix of shape (n_groups, N) as CSR.
    """
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    n_items = group_ids_arr.shape[0]
    row = group_ids_arr
    col = np.arange(n_items, dtype=np.int64)
    data = np.ones(n_items, dtype=np.float64)
    return cast(
        "csr_matrix",
        coo_matrix((data, (row, col)), shape=(n_groups, n_items)).tocsr(),
    )


def encode_dense_groups(
    group_ids: NDArray[np.integer],
    n_groups: int,
) -> NDArray[np.floating]:
    """Encode group IDs as a dense binary membership matrix.

    Args:
        group_ids: Integer group IDs of shape (N,).
        n_groups: Number of groups.

    Returns:
        Dense group membership matrix of shape (n_groups, N).
    """
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    n_items = group_ids_arr.shape[0]
    group_matrix = np.zeros((n_groups, n_items), dtype=np.float64)
    group_matrix[group_ids_arr, np.arange(n_items, dtype=np.int64)] = 1.0
    return group_matrix


def smooth(
    x: NDArray[np.floating],
    alpha: float = 0.02,
    out: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Apply simple smoothing along the last axis.

    Args:
        x: Input array; smoothing is applied along the last axis.
        alpha: Smoothing strength in [0, 1]. As alpha → 0 the output
            approaches the original data; as alpha → 1 the output
            approaches the mean along the last axis.
        out: Optional output array to write into. If provided, the smoothed
            values are written in-place and returned.

    Returns:
        Smoothed array with the same shape as x.
    """
    x_arr = np.asarray(x)
    smoothed = (1.0 - alpha) * x_arr + alpha * x_arr.mean(axis=-1, keepdims=True)
    if out is not None:
        np.copyto(out, smoothed)
        return out
    # Help mypy see this is an ndarray of floats.
    return cast("NDArray[np.floating]", smoothed)
