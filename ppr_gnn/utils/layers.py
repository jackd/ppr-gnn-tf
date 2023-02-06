import functools
import typing as tp

import gin
import tensorflow as tf

from . import ops

register = functools.partial(gin.register, module="ppr_gnn.utils.layers")


class DropoutV2(tf.keras.layers.Layer):
    """Dropout implementation that supports `SparseTensor` inputs."""

    def __init__(self, rate: float, seed: tp.Optional[int] = None, **kwargs):
        self.rate = rate
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(dict(rate=self.rate, seed=self.seed))
        return config

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        if training is None:
            training = tf.keras.backend.learning_phase()

        def train_fn():
            if isinstance(inputs, tf.SparseTensor):
                return ops.sparse_dropout(inputs, rate=self.rate, seed=self.seed)
            return tf.nn.dropout(inputs, rate=self.rate, seed=self.seed)

        def val_fn():
            return inputs

        return ops.smart_cond(training, train_fn, val_fn)


class Krylov(tf.keras.layers.Layer):
    """
    Create a Krylov subspace, `z_k = A**(k+1) @ x`.

    `k` and `axis` are specified at layer creation. `A` and `x` are provided during
    layer call.

    ```python
    layer = Krylov(k=3, axis=-2)
    z = layer((A, x))
    ```

    Where
        A: [..., N, N] coefficient matrix/tensor.
        x: [..., N, M]
        k: dimension of resulting space.
        axis: stacked dimension.

    z is the Krylov subspace with an extram `k`-sized dimension at `axis`,
        e.g. [..., N, k, M] for axis==-2.
    """

    def __init__(self, k: int, *, axis: int = -2, **kwargs):
        self.k = k
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(k=self.k, axis=self.axis)

    def call(self, inputs):
        A, x = inputs
        return ops.krylov(A, x, k=self.k, axis=self.axis)


def dense(x, units: int, **kwargs):
    """Create a keras `Dense` layer with (units, **kwargs) and apply it to `x`."""
    return tf.keras.layers.Dense(units=units, **kwargs)(x)


def dropout(
    x: tp.Union[tf.Tensor, tf.SparseTensor], dropout_rate: float
) -> tp.Union[tf.Tensor, tf.SparseTensor]:
    """Create a `DropoutV2` layer and apply it to `x`."""
    if dropout_rate:
        return DropoutV2(dropout_rate)(x)
    return x


@register
def batch_norm(x, **kwargs):
    """Create a keras `BatchNormalization` layer with **kwargs and apply it to `x`."""
    return tf.keras.layers.BatchNormalization(**kwargs)(x)


def krylov(A, x: tf.Tensor, k: int, axis: int = -2) -> tf.Tensor:
    """Wrapper around `Krylov` layer."""
    return Krylov(k=k, axis=axis)((A, x))
