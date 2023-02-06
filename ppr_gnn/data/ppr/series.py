import numpy as np
import scipy.sparse as sp
from tflo.matrix import (
    Matrix,
    ScaledIdentityMatrix,
    SparseMatrix,
    StaticPowerSeriesMatrix,
)

from ...utils import scipy_utils
from .base import register


@register
def ppr_maclaurin_series(
    A: sp.spmatrix,
    *,
    renormalize: bool = False,
    alpha: float = 0.1,
    k: int = 10,
    rescale: bool = False,
    dtype: np.dtype = np.float32
) -> Matrix:
    A = scipy_utils.sparse_normalize(A, renormalize=renormalize, dtype=dtype)
    n = A.shape[0]
    operator = SparseMatrix(
        scipy_utils.sp_to_tf(A), is_self_adjoint=True, is_positive_definite=True
    )
    operator = operator @ ScaledIdentityMatrix(n, 1 - alpha)
    coeffs = [1] * k
    mat = StaticPowerSeriesMatrix(operator, coeffs)
    if not rescale:
        return ScaledIdentityMatrix(n, alpha) @ mat
    return mat
