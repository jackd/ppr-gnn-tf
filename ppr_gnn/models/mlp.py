import typing as tp

import tensorflow as tf

from ..utils.layers import dense, dropout
from .core import register


@register
def mlp(
    features_spec: tf.TypeSpec,
    output_units: int,
    *,
    activation="relu",
    output_activation=None,
    hidden_units: tp.Iterable[int] = (64,),
    dropout_rate: float = 0.0,
    input_dropout_rate: tp.Optional[float] = None,
    l2_reg: float = 0.0,
    normalization: tp.Optional[tp.Callable] = None,
) -> tf.keras.Model:
    """
    Basic multi-layer perceptron.

    Each block is dense -> normalization -> activation -> dropout. There is an optional
    dropout initially, and no normalization/dropout applied after the final dense layer.

    Args:
        features_spec: `tf.TypeSpec` of input features.
        output_units: number of output units/channels/filters.
        *
        activation: activation function used between dense layers.
        output_activation: activation functions applied after final dense layer.
        hidden_units: number of units in each hidden dense layer.
        dropout_rate: used in intermediate dropouts.
        input_dropout_rate: rate used in dropout applied to the input. If `None`, uses
            `dropout_rate`.
        l2_reg: dense layers `use kernel_regularizer=tf.keras.regularizers.l2(l2_reg)`
        normalization: function used for normalization. This should create any necessary
            layers, e.g. `ppr_gnn.utils.layers.batch_norm`.

    Returns:
        `tf.keras.Model` which maps features with spec `features_spec` to tensor with
            shape `(*features_spec.shape[:-1], output_units)`.
    """
    activation = tf.keras.activations.get(activation)
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    inp = tf.keras.Input(type_spec=features_spec)

    x = dropout(inp, dropout_rate if input_dropout_rate is None else input_dropout_rate)
    for u in hidden_units:
        x = dense(x, units=u, kernel_regularizer=kernel_regularizer)
        if normalization:
            x = normalization(x)
        x = activation(x)
        x = dropout(x, dropout_rate)
    logits = dense(
        x,
        units=output_units,
        activation=output_activation,
        kernel_regularizer=kernel_regularizer,
    )

    return tf.keras.Model(inp, logits)
