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
        coeff: Physical diffusion coefficient D (units length^2 / time).
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

    The resulting operator corresponds to `coeff * Δ_h`, where `Δ_h` is the
    standard second-order central-difference Laplacian in 1D. No time-step
    scaling is applied here; `coeff` is interpreted as the physical diffusion
    coefficient `D` or a generic spatial scaling.

    For interior points, the stencil is:
        [1, -2, 1] / dx^2

    Boundary behavior depends on the `bc` argument.

    Args:
        n: Number of grid points.
        dx: Grid spacing.
        coeff: Physical diffusion coefficient D or a generic scaling factor.
        dtype: Floating dtype of the resulting matrix elements.
        bc: Boundary condition; either "neumann" or "absorbing".

    Returns:
        Scaled Laplacian matrix of shape (n, n) as a CSR sparse matrix.

    Raises:
        ValueError: If bc is not "neumann" or "absorbing".
    """
    dtype_obj = np.dtype(dtype)
    factor = coeff / dx**2

    main_diag = -2.0 * np.ones(n, dtype=dtype_obj)
    off_diag = np.ones(n - 1, dtype=dtype_obj)

    if bc == "neumann":
        main_diag[0] = -1.0
        main_diag[-1] = -1.0
    elif bc == "absorbing":
        main_diag[0] = -2.0
        main_diag[-1] = -2.0
    else:
        msg = UNKNOWN_BC_ERROR.format(bc=bc)
        raise ValueError(msg)

    # Use Python lists so SciPy's type stubs are happy.
    laplacian = diags(
        [off_diag.tolist(), main_diag.tolist(), off_diag.tolist()],
        [-1, 0, 1],
        shape=(n, n),
        dtype=dtype_obj,
    )

    scaled = laplacian * factor
    return scaled.tocsr()


def build_crank_nicolson_sparse(
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[csr_matrix, csr_matrix]:
    """Build sparse Crank-Nicolson operator matrices (L, R).

    This constructs the Crank-Nicolson time-stepping scheme for the linear
    diffusion equation

        u_t = D * Δ u

    using the discrete Laplacian Δ_h from `build_laplacian_tridiag`. The
    scheme is

        (I - 0.5 * dt * D * Δ_h) u^{n+1} = (I + 0.5 * dt * D * Δ_h) u^n.

    Args:
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration with coefficient D, dtype, and bc.
        dt: Time step size.

    Returns:
        A tuple (left_op, right_op) of CSR sparse matrices representing the
        left- and right-hand Crank-Nicolson operators.
    """
    n = geom.n
    dx = geom.dx
    coeff = cfg.coeff
    dtype_obj = np.dtype(cfg.dtype)
    bc = cfg.bc

    laplacian = build_laplacian_tridiag(
        n=n,
        dx=dx,
        coeff=coeff,
        dtype=dtype_obj,
        bc=bc,
    )

    time_scaled_op = laplacian * dt

    identity_mat = identity(
        n,
        dtype=dtype_obj,
        format="csr",
    )
    left_op = identity_mat - 0.5 * time_scaled_op
    right_op = identity_mat + 0.5 * time_scaled_op

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
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build dense Crank-Nicolson operator matrices (L, R).

    This constructs the Crank-Nicolson time-stepping scheme for the linear
    diffusion equation

        u_t = D * Δ u

    using the discrete Laplacian Δ_h from `build_laplacian_tridiag`. The
    scheme is

        (I - 0.5 * dt * D * Δ_h) u^{n+1} = (I + 0.5 * dt * D * Δ_h) u^n.

    Args:
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration with coefficient D, dtype, and bc.
        dt: Time step size.

    Returns:
        A tuple (left_op, right_op) of dense arrays representing the left- and
        right-hand Crank-Nicolson operators.
    """
    n = geom.n
    dx = geom.dx
    coeff = cfg.coeff
    dtype_obj = np.dtype(cfg.dtype)
    bc = cfg.bc

    laplacian = build_laplacian_tridiag(
        n=n,
        dx=dx,
        coeff=coeff,
        dtype=dtype_obj,
        bc=bc,
    ).toarray()

    time_scaled_op = dt * laplacian

    identity_mat = np.eye(n, dtype=dtype_obj)
    left_op = identity_mat - 0.5 * time_scaled_op
    right_op = identity_mat + 0.5 * time_scaled_op

    if bc == "absorbing":
        for mat in (left_op, right_op):
            mat[0, :] = 0.0
            mat[0, 0] = 1.0
            mat[-1, :] = 0.0
            mat[-1, -1] = 1.0

    return left_op, right_op


def build_crank_nicolson_operator(
    geom: GridGeometry,
    cfg: DiffusionConfig,
    dt: float,
) -> tuple[Any, Any]:
    """Build Crank-Nicolson operators with dense/sparse autodispatch.

    Small systems are built as dense operators; large systems as sparse.
    The decision is based on the grid size `geom.n` and the internal
    `_DISPATCH_THRESHOLD`.

    Args:
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration with coefficient D, dtype, and bc.
        dt: Time step size.

    Returns:
        A tuple (left_op, right_op) of operator matrices, either dense arrays
        or CSR sparse matrices depending on problem size.
    """
    n = geom.n
    if n < _DISPATCH_THRESHOLD:
        return build_crank_nicolson_dense(geom, cfg, dt)
    return build_crank_nicolson_sparse(geom, cfg, dt)


# --- Predictor-Corrector matrices ---


def build_predictor_corrector_dense(
    base_matrix: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Build predictor-corrector matrices for a dense base matrix.

    This constructs matrices suitable for an IMEX-style predictor-corrector
    scheme where the linear part is handled implicitly via a
    Crank-Nicolson-like step, and nonlinear/stochastic reaction terms are
    handled explicitly at a higher level.

    The input is a time-scaled linear operator

        base_matrix ≈ dt * A,

    where `A` might be a diffusion operator (e.g., `D * Δ_h`) or a more
    general linear coupling.

    The matrices are:

        predictor = I
        left_op   = I - 0.5 * base_matrix
        right_op  = I + 0.5 * base_matrix

    so that a single implicit linear step can be written as

        left_op @ y^{n+1} = right_op @ y^{n} + (reaction terms).

    Any explicit predictor step for the reaction part (e.g. Euler or Heun)
    should be implemented outside this function.

    Args:
        base_matrix: Time-scaled linear operator of shape (n, n), typically
            base_matrix = dt * A.

    Returns:
        A tuple (predictor, left_op, right_op) of dense matrices for use in a
        predictor-corrector scheme, where predictor is the identity and
        left_op/right_op encode the Crank-Nicolson treatment of the linear
        operator.
    """
    n = base_matrix.shape[0]
    identity_mat = np.eye(n, dtype=base_matrix.dtype)
    predictor = identity_mat
    left_op = identity_mat - 0.5 * base_matrix
    right_op = identity_mat + 0.5 * base_matrix
    return predictor, left_op, right_op


def build_predictor_corrector_sparse(
    base_matrix: csr_matrix,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build predictor-corrector matrices for a sparse base matrix.

    This constructs matrices suitable for an IMEX-style predictor-corrector
    scheme where the linear part is handled implicitly via a
    Crank-Nicolson-like step, and nonlinear/stochastic reaction terms are
    handled explicitly at a higher level.

    The input is a time-scaled linear operator

        base_matrix ≈ dt * A,

    where `A` might be a diffusion operator (e.g., `D * Δ_h`) or a more
    general linear coupling.

    The matrices are:

        predictor = I
        left_op   = I - 0.5 * base_matrix
        right_op  = I + 0.5 * base_matrix

    so that a single implicit linear step can be written as

        left_op @ y^{n+1} = right_op @ y^{n} + (reaction terms).

    Any explicit predictor step for the reaction part (e.g. Euler or Heun)
    should be implemented outside this function.

    Args:
        base_matrix: Time-scaled linear operator as a CSR sparse matrix of
            shape (n, n), typically base_matrix = dt * A.

    Returns:
        A tuple (predictor, left_op, right_op) of CSR sparse matrices for use
        in a predictor-corrector scheme, where predictor is the identity and
        left_op/right_op encode the Crank-Nicolson treatment of the linear
        operator.
    """
    n = base_matrix.shape[0]
    identity_mat = identity(
        n,
        format="csr",
        dtype=np.float64,
    )
    predictor = identity_mat
    left_op = identity_mat - 0.5 * base_matrix
    right_op = identity_mat + 0.5 * base_matrix
    return (
        cast("csr_matrix", predictor.tocsr()),
        cast("csr_matrix", left_op.tocsr()),
        cast("csr_matrix", right_op.tocsr()),
    )


def build_predictor_corrector(
    base_matrix: NDArray[np.floating] | csr_matrix,
) -> tuple[Any, Any, Any]:
    """Build predictor-corrector matrices with dense/sparse autodispatch.

    This function accepts either a dense or sparse time-scaled linear operator

        base_matrix ≈ dt * A,

    and constructs predictor-corrector matrices in a dense or sparse format
    depending on the input type and problem size.

    The returned matrices are:

        predictor = I
        left_op   = I - 0.5 * base_matrix
        right_op  = I + 0.5 * base_matrix

    so that the linear part can be advanced implicitly in a Crank-Nicolson
    fashion, while explicit predictor/corrector steps for nonlinear/stochastic
    reaction terms are handled by higher-level solver logic.

    Args:
        base_matrix: Time-scaled linear operator, either a dense array-like or
            a CSR sparse matrix.

    Returns:
        A tuple (predictor, left_op, right_op) where the matrices are dense or
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

    The returned callable solves the linear system

        left_op @ y = right_op @ x

    for y, given x. It uses LU factorization for dense matrices and
    sparse factorization for CSR matrices. The choice of sparse vs dense
    is based on the types of `left_op` and `right_op`.

    Args:
        left_op: Left-hand operator matrix L (dense or sparse).
        right_op: Right-hand operator matrix R (dense or sparse).

    Returns:
        A callable solver(x) that returns the solution y to
        left_op @ y = right_op @ x.
    """
    # Only treat as sparse if both operators are sparse.
    is_sparse = issparse(left_op) and issparse(right_op)

    if is_sparse:
        left_csr = cast("csr_matrix", left_op)
        right_csr = cast("csr_matrix", right_op)

        solve_left = sparse_factorized(left_csr)

        def solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
            """Solve left_op @ y = right_op @ x for sparse operators.

            Args:
                x: Right-hand side vector or matrix.

            Returns:
                Solution array y with the same shape as x.
            """
            x_arr = np.asarray(x, dtype=left_csr.dtype)
            rhs = right_csr @ x_arr
            return np.asarray(solve_left(rhs), dtype=x_arr.dtype)

    else:
        left_dense = np.asarray(left_op)
        right_dense = np.asarray(right_op)
        lu, piv = lu_factor(left_dense)

        def solver(x: NDArray[np.floating]) -> NDArray[np.floating]:
            """Solve left_op @ y = right_op @ x for dense operators.

            Args:
                x: Right-hand side vector or matrix.

            Returns:
                Solution array y with the same shape as x.
            """
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

    This solves the linear system

        left_op @ y = right_op @ x

    for y, reusing cached factorizations keyed by the identities of
    `left_op` and `right_op`. This allows efficient repeated solves
    across timesteps or subgroups.

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
    dt: float,
) -> tuple[str, tuple[Any, ...]]:
    """Dispatch solver operator construction based on rule structure.

    This function chooses between Crank-Nicolson and a predictor-corrector
    scheme based on the presence of nonlinear or stochastic rules. Time-step
    scaling is handled explicitly via the `dt` argument, while `cfg.coeff`
    is always interpreted as the physical diffusion coefficient D.

    If any rule is nonlinear or stochastic, a predictor-corrector scheme
    is used. Otherwise, a pure Crank-Nicolson scheme is chosen.

    Args:
        rules: List of rule objects; each must expose boolean attributes
            `nonlinear` and `stochastic`.
        geom: Grid geometry containing grid size and spacing.
        cfg: Diffusion configuration containing coefficient D, dtype, and
            boundary condition.
        dt: Time step size.

    Returns:
        A tuple (method, operators) where:

        * method is either "crank-nicolson" or "predictor-corrector".
        * operators is:
            * (left_op, right_op) for Crank-Nicolson.
            * (predictor, left_op, right_op) for predictor-corrector.
    """
    n = geom.n
    dx = geom.dx
    coeff = cfg.coeff
    dtype_obj = np.dtype(cfg.dtype)
    bc = cfg.bc

    has_nonlinear = any(getattr(rule, "nonlinear", False) for rule in rules)
    has_stochastic = any(getattr(rule, "stochastic", False) for rule in rules)

    if has_nonlinear or has_stochastic:
        laplacian = build_laplacian_tridiag(
            n=n,
            dx=dx,
            coeff=coeff,
            dtype=dtype_obj,
            bc=bc,
        )

        time_scaled_op = laplacian * dt
        predictor, left_op, right_op = build_predictor_corrector(time_scaled_op)
        operators: tuple[Any, ...] = (predictor, left_op, right_op)
        method = "predictor-corrector"
    else:
        left_op, right_op = build_crank_nicolson_operator(
            geom=geom,
            cfg=cfg,
            dt=dt,
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

    Performs the matrix multiplication

        (G x N) @ (N, ...) -> (G, ...),

    where G is the number of groups and N is the number of items.

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

    Performs the matrix multiplication

        (G x N) @ (N, ...) -> (G, ...),

    where G is the number of groups and N is the number of items.

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

    This selects an efficient dense or sparse implementation for computing
    grouped sums using a group membership matrix.

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

    This computes the size of each group represented in a sparse group
    membership matrix.

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

    This computes the size of each group represented in a dense group
    membership matrix.

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

    This selects an efficient dense or sparse implementation for computing
    group sizes using a group membership matrix.

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

    This computes aggregated values for arbitrary masks encoded as rows
    of a sparse mask matrix.

    Args:
        mask_matrix: Sparse mask matrix of shape (M, N).
        data: Data to aggregate of shape (N,) or (N, K).

    Returns:
        Masked sum result as an array of shape (M,) or (M, K), depending
        on the shape of `data`.
    """
    return np.asarray(mask_matrix @ data)


def matrix_masked_sum_dense(
    mask_matrix: NDArray[np.floating],
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums using a dense mask matrix.

    This computes aggregated values for arbitrary masks encoded as rows
    of a dense mask matrix.

    Args:
        mask_matrix: Dense mask matrix of shape (M, N).
        data: Data to aggregate of shape (N,) or (N, K).

    Returns:
        Masked sum result as an array of shape (M,) or (M, K), depending
        on the shape of `data`.
    """
    return np.asarray(mask_matrix @ data)


def matrix_masked_sum(
    mask_matrix: csr_matrix | NDArray[np.floating],
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute masked sums with dense/sparse autodispatch.

    This selects an efficient dense or sparse implementation for computing
    masked sums using a mask matrix.

    Args:
        mask_matrix: Mask matrix, dense or sparse, of shape (M, N).
        data: Data to aggregate of shape (N,) or (N, K).

    Returns:
        Masked sum result as an array with shape determined by the mask and
        data, typically (M,) or (M, K).
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

    This computes group sizes using a fast bincount-based implementation.

    Args:
        group_ids: Integer group IDs of shape (N,) in the range
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

    This computes group-wise sums using integer group IDs and a fast
    bincount-based implementation.

    Args:
        values: Values to sum of shape (N,).
        group_ids: Integer group IDs of shape (N,) in the range
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

    This computes group-wise sums along the first axis of `values` for
    each feature, using integer group IDs and a fast bincount-based
    implementation.

    Args:
        values: Values to sum of shape (N, K) (e.g., compartments per
            individual).
        group_ids: Integer group IDs of shape (N,) in the range
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

    This creates a sparse binary matrix of shape (n_groups, N) where each
    column corresponds to an item and has a single 1.0 entry in the row
    corresponding to its group ID.

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

    This creates a dense binary matrix of shape (n_groups, N) where each
    column corresponds to an item and has a single 1.0 entry in the row
    corresponding to its group ID.

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

    This applies an exponential-like smoothing that blends each value with
    the mean along the last axis:

        smoothed = (1 - alpha) * x + alpha * mean(x, axis=-1, keepdims=True)

    As `alpha → 0`, the output approaches the original data; as `alpha → 1`,
    the output approaches the mean along the last axis.

    Args:
        x: Input array; smoothing is applied along the last axis.
        alpha: Smoothing strength in [0, 1].
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
