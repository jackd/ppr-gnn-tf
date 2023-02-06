import functools

import gacl
import tensorflow as tf
from absl import app, flags

from ppr_gnn.data import transforms
from ppr_gnn.data.ppr.cg import ppr_cg
from ppr_gnn.data.transitive import (
    get_data,
    propagate_mlp_split,
    transform_transitive_data,
)
from ppr_gnn.models import mlp_propagate
from ppr_gnn.utils.callbacks import get_model_callbacks
from ppr_gnn.utils.experiment_callbacks import FitReporter, KerasSeedSetter
from ppr_gnn.utils.train import build_fit_test

flags.DEFINE_string("data", "cora", "base dataset to use")

flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_integer("trials", 10, "number of trials to run")
flags.DEFINE_integer("epochs", 1500, "number of steps to train for")
flags.DEFINE_integer("patience", 100, "early stopping patience")
flags.DEFINE_integer(
    "max_iter", 1000, "maximum iterations used in conjugate gradient solver"
)

flags.DEFINE_bool("renormalize", True, "add self loops before normalization")
flags.DEFINE_bool("lc", False, "Extract the graph's largest component")
flags.DEFINE_bool("rescale", True, "Use rescaled PPR")
flags.DEFINE_bool("gcn2", False, "Use GCN2 architecture, otherwise use DAGNN")

flags.DEFINE_float("dropout", 0.8, "dropout_rate")
flags.DEFINE_float("l2", 2.5e-3, "l2 regularization weight")
flags.DEFINE_float("alpha", 0.1, "alpha value used in PPR")
flags.DEFINE_float("tol", 1e-3, "tolerance used in conjugate gradient solver")
flags.DEFINE_float("lr", 1e-2, "optimizer learning rate")


FLAGS = flags.FLAGS


def run():
    alpha = FLAGS.alpha
    max_iter = FLAGS.max_iter
    tol = FLAGS.tol
    rescale = FLAGS.rescale
    largest_component_only = FLAGS.lc
    renormalize = FLAGS.renormalize
    data_name = FLAGS.data
    epochs = FLAGS.epochs
    patience = FLAGS.patience
    dropout_rate = FLAGS.dropout
    l2_reg = FLAGS.l2
    lr = FLAGS.lr

    monitor = "val_cross_entropy"
    mode = "min"

    data = get_data(data_name)
    data = transform_transitive_data(
        data,
        features_transform=transforms.row_normalize,
        adjacency_transform=functools.partial(
            transforms.sparse_normalize, renormalize=renormalize
        ),
        largest_component_only=largest_component_only,
    )
    split = propagate_mlp_split(
        data,
        propagator_fn=functools.partial(
            ppr_cg,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            rescale=rescale,
        ),
    )
    num_classes = data.labels.max() + 1

    callbacks = get_model_callbacks(monitor=monitor, mode=mode, patience=patience)

    model_fn = functools.partial(
        mlp_propagate,
        num_classes=num_classes,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        version="gcn2" if FLAGS.gcn2 else "dagnn",
    )

    optimizer = tf.keras.optimizers.Adam(lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    weighted_metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=True, name="cross_entropy"
        ),
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ]

    return build_fit_test(
        split=split,
        model_fn=model_fn,
        epochs=epochs,
        callbacks=callbacks,
        optimizer=optimizer,
        loss=loss,
        weighted_metrics=weighted_metrics,
    )


def main(argv=None):
    del argv
    experiment_callbacks = [
        KerasSeedSetter(FLAGS.seed),
        FitReporter(),
    ]
    gacl.main(run, callbacks=experiment_callbacks, num_trials=FLAGS.trials)


if __name__ == "__main__":
    app.run(main)
