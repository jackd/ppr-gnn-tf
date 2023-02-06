import typing as tp

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tflo.matrix import Matrix, ScaledIdentityMatrix, SparseMatrix
from tflo.matrix.core import register_matrix_cls

from ...utils import scipy_utils
from .base import register


def _ppr_recursive(fn: tp.Callable, x0: tf.Tensor, alpha: float, k: int) -> tf.Tensor:
    x = x0
    for _ in range(k):
        x = (1 - alpha) * fn(x) + alpha * x0
    return x


class LinearOperatorPPRRecursive(
    tf.linalg.LinearOperator
):  # pylint: disable=abstract-method
    """
    LinearOperator representing recursive Personalized PageRank matrix.

    Updates via
    $X^{(k+1)} = (1 - alpha)X^{(k)} + alpha X^{(0)}$
    """

    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        alpha: float,
        k: int,
        name="LinearOperatorPPR",
    ):
        assert operator.is_square
        self.operator = operator
        self.alpha = alpha
        self.k = k
        super().__init__(
            dtype=operator.dtype,
            is_self_adjoint=operator.is_self_adjoint,
            parameters=dict(operator=operator, alpha=alpha, k=k),
            name=name,
        )

    def _shape(self):
        return self.operator.shape

    def _shape_tensor(self):
        return self.operator.shape_tensor()

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint_arg:
            x = tf.math.conj(tf.transpose(x))
        return _ppr_recursive(
            lambda x: self.operator.matmul(x, adjoint=adjoint), x, self.alpha, self.k
        )

    def _matvec(self, x, adjoint: bool = False):
        return _ppr_recursive(
            lambda x: self.operator.matvec(x, adjoint=adjoint), x, self.alpha, self.k
        )

    @property
    def _composite_tensor_fields(self):
        return ("operator", "alpha", "k")

    def _adjoint(self) -> "LinearOperatorPPRRecursive":
        return LinearOperatorPPRRecursive(
            self.operator.adjoint(), alpha=self.alpha, k=self.k
        )


@register_matrix_cls(LinearOperatorPPRRecursive)
class PPRRecursiveMatrix(Matrix):  # pylint: disable=abstract-method
    operator: Matrix
    alpha: float
    k: int
    name: str = "ExponentialMatrix"

    class Spec:
        @property
        def shape(self):
            return self.operator.shape  # pylint: disable=no-member

        @property
        def dtype(self):
            return self.operator.dtype  # pylint: disable=no-member


@register
def ppr_recursive(
    A: sp.spmatrix,
    *,
    renormalize: bool = False,
    alpha: float = 0.1,
    k: int = 10,
    rescale: bool = False,
    dtype: np.dtype = np.float32
) -> Matrix:
    A = scipy_utils.sparse_normalize(A, dtype=dtype, renormalize=renormalize)
    mat = PPRRecursiveMatrix(SparseMatrix(scipy_utils.sp_to_tf(A)), alpha=alpha, k=k)
    if rescale:
        return ScaledIdentityMatrix(mat.shape[0], 1.0 / alpha) @ mat
    return mat
