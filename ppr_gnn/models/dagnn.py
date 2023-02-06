import typing as tp

import tensorflow as tf

from ..utils.layers import dense, dropout, krylov
from .core import register


@register
def dagnn(
    input_spec: tp.Tuple[tf.TypeSpec, tf.TypeSpec],
    num_classes: int,
    *,
    hidden_size: int = 256,
    input_dropout_rate: tp.Optional[float] = None,
    dropout_rate: float = 0.2,
    num_propagations: int = 16,
    l2_reg: float = 0.0,
    static: int = False,
) -> tf.keras.Model:
    """
    DAGNN model constructor.

    ```bibtex
    @inproceedings{liu2020towards,
        title={Towards Deeper Graph Neural Networks},
        author={Liu, Meng and Gao, Hongyang and Ji, Shuiwang},
        booktitle={Proceedings of the 26th ACM SIGKDD International Conference on
                   Knowledge Discovery \& Data Mining},
        year={2020},
        organization={ACM}
    }
    ```

    Args:
        input_spec: `tf.TypeSpec`s of (features, simple_propagator) inputs/
        num_classes: number of output units/channels/filters.
        hidden_size: number of hidden filters.
        input_dropout_rate: rate used in dropout for inputs. Uses `dropout_rate` if
            `None`.
        dropout_Rate: rate used in dropout after initial dropout.
        num_propagations: number of propagations used in propagation.
        l2_reg: l2 regularization weight.
        static: if True, uses static propagation (used in SS-DAGNN).

    Returns:
        Uncompiled `tf.keras.Model` mapping `inputs -> logits`.
    """
    kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    inputs = tf.nest.map_structure(lambda s: tf.keras.Input(type_spec=s), input_spec)
    x, A = inputs

    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    kwargs = dict(
        kernel_regularizer=reg,
        bias_regularizer=reg,
    )

    x = dropout(x, dropout_rate if input_dropout_rate is None else input_dropout_rate)
    x = tf.keras.layers.Dense(
        hidden_size,
        activation="relu",
        **kwargs,
    )(x)
    x = dropout(x, dropout_rate)
    x = tf.keras.layers.Dense(num_classes, **kwargs)(x)
    terms = krylov(A, x, k=num_propagations + 1, axis=-2)  # [N, k, num_classes]
    if static:
        logits = tf.reduce_sum(terms, axis=-2)
    else:
        adaptive_factor = dense(
            terms, 1, kernel_regularizer=kernel_regularizer, activation=tf.nn.sigmoid
        )
        adaptive_factor = tf.squeeze(adaptive_factor, axis=-1)  # [N, k]
        logits = tf.linalg.matvec(terms, adaptive_factor, transpose_a=True)

    model = tf.keras.Model(inputs, logits)
    return model
