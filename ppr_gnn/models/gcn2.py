import typing as tp

import numpy as np
import tensorflow as tf

from ..utils.layers import dense, dropout
from .core import register


def _matmul(A: tp.Union[tf.SparseTensor, tf.Tensor], X: tf.Tensor) -> tf.Tensor:
    if isinstance(A, tf.SparseTensor):
        return tf.sparse.sparse_dense_matmul(A, X)
    return A @ X


class StaticGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, alpha: float, beta: float, *, activation=None, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            alpha=self.alpha,
            beta=self.beta,
            activation=tf.keras.utils.serialize_keras_object(self.activation),
        )
        return config

    def call(self, inputs):
        features, features0, simple_prop = inputs
        hi = _matmul(simple_prop, features)
        x = (1 - self.alpha) * hi + self.alpha * features0
        return self.activation(x)


class DynamicGraphConvolution(tf.keras.layers.Layer):
    def __init__(
        self,
        alpha: float,
        beta: float,
        *,
        variant: bool = False,
        kernel_regularizer: tp.Optional[tf.keras.regularizers.Regularizer] = None,
        activation=None,
        **kwargs,
    ):
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            alpha=self.alpha,
            beta=self.beta,
            variant=self.variant,
            kernel_regularizer=tf.keras.utils.serialize_keras_object(
                self.kernel_regularizer
            ),
            activation=tf.keras.utils.serialize_keras_object(self.activation),
        )
        return config

    def build(self, input_shape):
        if self.built:
            return
        super().build(input_shape)
        filters = input_shape[0][-1]
        self._dense = tf.keras.layers.Dense(
            filters, kernel_regularizer=self.kernel_regularizer, use_bias=False
        )
        self._dense.build(input_shape[0])

    def call(self, inputs):
        features, features0, simple_prop = inputs
        hi = _matmul(simple_prop, features)

        if self.variant:
            support = tf.concat([hi, features0], axis=1)
            r = (1 - self.alpha) * hi + self.alpha * features
        else:
            support = (1 - self.alpha) * hi + self.alpha * features0
            r = support

        output = self.beta * self._dense(support) + (1 - self.beta) * r
        return self.activation(output)


def _graph_conv(
    adj: tp.Union[tf.Tensor, tf.SparseTensor],
    features: tf.Tensor,
    features0: tf.Tensor,
    alpha: float,
    beta: float,
    variant: bool,
    l2_reg: float,
    activation: tp.Callable,
    name: str,
    static: bool,
):
    shared_kwargs = dict(alpha=alpha, beta=beta, name=name, activation=activation)
    if static:
        layer = StaticGraphConvolution(**shared_kwargs)
    else:
        layer = DynamicGraphConvolution(
            variant=variant,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            **shared_kwargs,
        )
    return layer((features, features0, adj))


@register
def gcn2(
    inputs_spec: tp.Tuple[tf.TypeSpec, tf.TypeSpec],
    num_classes: int,
    *,
    filters: int = 64,
    num_hidden_layers: int = 64,
    dropout_rate: float = 0.6,
    lam: float = 0.5,
    alpha: float = 0.1,
    variant: bool = False,
    activation: tp.Union[str, tp.Callable] = "relu",
    conv_l2_reg: float = 0.0,
    dense_l2_reg: float = 0.0,
    static: bool = False,
):
    """
    GCN2 model constructor.

    ```bibtex
    @article{chenWHDL2020gcnii,
        title = {Simple and Deep Graph Convolutional Networks},
        author = {Ming Chen, Zhewei Wei and Zengfeng Huang, Bolin Ding and Yaliang Li},
        year = {2020},
        booktitle = {Proceedings of the 37th International Conference on Machine
                     Learning},
    }
    ```

    Args:
        inputs_spec: (features, simple_propagator) `tf.TypeSpec`s.
        num_classes: number of output units/filters/channels.
        filters: number of units per hidden layer.
        num_hidden_layers: number of hidden GCN layers.
        dropout_rate: used in dropout layers.
        lam, alpha, variant: see original GCN2 paper.
        activation: activation function applied between layers.
        conv_l2_reg: l2 regularization weight applied to GCN kernels.
        dense_l2_reg: l2 regularization weight applied to first and last dense kernels.
        static: if True, uses SS-GCN2 architecture.

    Returns:
        Uncompiled `tf.keras.Model` mapping `inputs -> logits`.
    """
    activation = tf.keras.activations.get(activation)

    kernel_regularizer = tf.keras.regularizers.l2(dense_l2_reg)

    inputs = tf.nest.map_structure(
        lambda spec: tf.keras.Input(type_spec=spec), inputs_spec
    )
    x, adjacency = inputs
    x = dropout(x, dropout_rate)
    x = dense(
        x,
        filters,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        name="linear_0",
    )
    x0 = x
    for i in range(num_hidden_layers):
        x = dropout(x, dropout_rate)
        x = _graph_conv(
            adjacency,
            x,
            x0,
            alpha=alpha,
            beta=np.log(lam / (i + 1) + 1),
            variant=variant,
            l2_reg=conv_l2_reg,
            activation=activation,
            name=f"conv_{i}",
            static=static,
        )

    x = dropout(x, dropout_rate)
    x = dense(x, num_classes, kernel_regularizer=kernel_regularizer, name="linear_1")
    return tf.keras.Model(inputs, x)
