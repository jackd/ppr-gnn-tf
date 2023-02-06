import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf
from tflo.matrix import FullMatrix, Matrix

from ...utils import scipy_utils
from .base import register


@register
def ppr_low_rank(
    A: sp.spmatrix,
    *,
    renormalize: bool = False,
    alpha: float = 0.1,
    rank: int = 100,
    rescale: bool = False,
    dtype: np.dtype = np.float32
) -> Matrix:
    A = scipy_utils.sparse_normalize(A, dtype=dtype, renormalize=renormalize)
    n = A.shape[0]
    coo = -sp.eye(n, dtype=dtype) - A  # L - 2I
    w, v = la.eigsh(coo, k=rank, which="LM")
    w += alpha + 2
    if not rescale:
        w = w / alpha
    w = tf.convert_to_tensor(w, dtype=dtype)
    v = tf.convert_to_tensor(v, dtype=dtype)
    return tf.linalg.matmul(FullMatrix(v / w), FullMatrix(v), adjoint_b=True)
