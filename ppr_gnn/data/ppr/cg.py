import functools
import typing as tp

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from tflo.matrix import CGSolverMatrix, Matrix, ScaledIdentityMatrix, SparseMatrix

from ...utils import scipy_utils
from .base import register


def _scaled_ppr_inverse_preprocess(
    A: sp.spmatrix,
    alpha: float,
    renormalize: bool = False,
    dtype: np.dtype = np.float32,
) -> tp.Tuple[sp.spmatrix, np.ndarray]:
    """
    Perform preprocessing for ppr inverse preprocessing.

    Returns:
        operator: I - (1 - alpha) A_hat
        x0: normalized np.sqrt(d)
    """
    assert sp.issparse(A), A
    if A.dtype != dtype:
        A = A.astype(dtype)
    n = A.shape[0]
    I = sp.eye(n, dtype=dtype)
    if renormalize:
        A = A + I

    d = np.array(A.sum(axis=1)).squeeze(1)
    d_sqrt = np.sqrt(d)
    d_sqrt[d == 0] = 0
    A = A.tocoo()

    data = A.data / (d_sqrt[A.row] * d_sqrt[A.col]) * (alpha - 1)
    operator = sp.coo_matrix((data, (A.row, A.col)), shape=A.shape)
    operator = I + operator

    x0 = d_sqrt
    x0 /= np.linalg.norm(x0)
    return operator, x0


@register
def ppr_cg(
    A: sp.spmatrix,
    *,
    renormalize: bool = False,
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-3,
    rescale: bool = False,
    dtype: np.dtype = np.float32,
) -> Matrix:
    operator, x0 = _scaled_ppr_inverse_preprocess(
        A, alpha, renormalize=renormalize, dtype=dtype
    )

    operator = SparseMatrix(
        scipy_utils.sp_to_tf(operator), is_self_adjoint=True, is_positive_definite=True
    )
    mat = CGSolverMatrix(
        operator,
        tol=tol,
        max_iter=max_iter,
        x0=x0,
    )
    if not rescale:
        return ScaledIdentityMatrix(mat.shape[0], alpha) @ mat
    return mat


def _rescale_wrapper(fn, factor: float):
    def ret_fn(x):
        return factor * fn(x)

    return ret_fn


def _cg(*args, **kwargs):
    x, _ = la.cg(*args, **kwargs)
    return x


@register
def ppr_cg_np(
    A: sp.spmatrix,
    *,
    renormalize: bool = False,
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-3,
    rescale: bool = False,
    dtype: np.dtype = np.float32,
) -> tp.Callable[[np.ndarray], np.ndarray]:
    operator, x0 = _scaled_ppr_inverse_preprocess(
        A, alpha, renormalize=renormalize, dtype=dtype
    )
    base_fn = functools.partial(_cg, operator, maxiter=max_iter, tol=tol, x0=x0)
    if not rescale:
        return _rescale_wrapper(base_fn, alpha)
    return base_fn
