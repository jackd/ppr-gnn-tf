import functools
import typing as tp

import tensorflow as tf
from tflo.matrix import Matrix, SparseMatrix


def _sparse_dropout(x: tf.SparseTensor, rate: float, uniform_fn: tp.Callable):
    assert x.dtype.is_floating
    mask = uniform_fn(tf.shape(x.values), dtype=tf.float32) >= rate
    st = tf.sparse.retain(x, mask)
    # gradient issues with automatic broadcasting of sparse tensors
    # https://github.com/tensorflow/tensorflow/issues/46008#issuecomment-751755570
    out = st.with_values(st.values / (1 - rate))
    return out


def sparse_dropout(x: tf.SparseTensor, rate: float, seed=None) -> tf.SparseTensor:
    """Apply `dropout` to a `tf.SparseTensor`."""
    return _sparse_dropout(x, rate, functools.partial(tf.random.uniform, seed=seed))


def smart_cond(
    cond: tp.Union[bool, tf.Tensor, tf.Variable],
    if_true: tp.Callable,
    if_false: tp.Callable,
):
    """Conditional branch that calls if_true or if_false directly if cond is a bool."""
    if isinstance(cond, (tf.Tensor, tf.Variable)):
        assert cond.dtype.is_bool
        return tf.cond(cond, if_true, if_false)
    if cond:
        return if_true()
    return if_false()


def krylov(
    A: tp.Union[tf.Tensor, tf.SparseTensor, Matrix],
    x: tf.Tensor,
    k: int,
    axis: int = -2,
) -> tf.Tensor:
    """
    Create a Krylov subspace, `z_k = A**k @ x`.

    Args:
        A: [..., N, N] coefficient matrix/tensor.
        x: [..., N, M]
        k: dimension of resulting space.
        axis: stacked dimension.

    Returns:
        Krylov subspace with an extram `k`-sized dimension at `axis`,
            e.g. [..., N, k, M] for axis==-2
    """
    if isinstance(tf.type_spec_from_value(A), tf.SparseTensorSpec):
        A = SparseMatrix(A)
    z = [x]
    for _ in range(k - 1):
        z.append(A @ z[-1])
    return tf.stack(z, axis=axis)
