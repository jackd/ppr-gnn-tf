import typing as tp

import tensorflow as tf

from ..utils.layers import dense, dropout
from .core import register


class MLPPropagateVersions:
    DAGNN = "dagnn"
    GCN2 = "gcn2"


@register
def mlp_propagate(
    input_spec: tp.Tuple[tf.TypeSpec, tf.TypeSpec],
    num_classes: int,
    *,
    activation: tp.Union[tp.Callable, str] = "relu",
    output_activation=None,
    hidden_units: tp.Union[int, tp.Iterable[int]] = (64,),
    dropout_rate: float = 0.0,
    input_dropout_rate: tp.Optional[float] = None,
    l2_reg: float = 0.0,
    normalization: tp.Optional[tp.Callable] = None,
    version: str = MLPPropagateVersions.DAGNN,
) -> tf.keras.Model:
    """
    MLP-propagate model.

    Args:
        input_spec: `tf.TypeSpec`s of (features, propagator) inputs.
        num_classes: number of output units/filters/channels.
        activation: non-linearity applied in MLP.
        output_activation: output non-linearity.
        hidden_units: iterable of sizes of intermediate MLP layers.
        dropout_rate: rate used in intermediate dropout layers.
        input_dropout_rate: rate used in initial dropout layer. If `None`, uses
            `dropout_rate`.
        l2_reg: l2 regularization weight.
        normalization: normalization functions applied between dense layers of MLP.
        version: one of "dagnn" or "gcn2" to change between macro architectures.

    Returns:
        Uncompiled `tf.keras.Model` mapping `inputs -> logits`.
    """
    activation = tf.keras.activations.get(activation)
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)

    if not hasattr(hidden_units, "__iter__"):
        hidden_units = (hidden_units,)
    inp = tf.nest.map_structure(lambda spec: tf.keras.Input(type_spec=spec), input_spec)
    x, prop = inp

    x = dropout(
        x,
        dropout_rate=dropout_rate if input_dropout_rate is None else input_dropout_rate,
    )
    for units in hidden_units:
        x = dense(x, units=units, kernel_regularizer=kernel_regularizer)
        if normalization:
            x = normalization(x)
        x = activation(x)
        x = dropout(x, dropout_rate)
    output_kwargs = dict(
        activation=output_activation,
        units=num_classes,
        kernel_regularizer=kernel_regularizer,
    )
    if version == MLPPropagateVersions.DAGNN:
        logits = prop @ dense(x, **output_kwargs)
    else:
        assert version == MLPPropagateVersions.GCN2, version
        logits = dense(dropout(prop @ x, dropout_rate), **output_kwargs)
    return tf.keras.Model(inp, logits)
