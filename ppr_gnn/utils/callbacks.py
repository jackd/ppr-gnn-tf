import functools
import sys
import typing as tp

import gin
import tensorflow as tf

register = functools.partial(gin.register, module="ppr_gnn.utils.callbacks")


class EpochProgbarLogger(tf.keras.callbacks.Callback):
    """Progress bar that updates at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.progbar = None
        self.epochs = None
        self.last_seen = None

    def set_params(self, params):
        self.epochs = params["epochs"]

    def on_train_begin(self, logs=None):
        del logs

        class Universe:
            """Contains everything."""

            def __contains__(self, x):
                return True

        self.progbar = tf.keras.utils.Progbar(
            target=self.epochs,
            unit_name="epoch",
        )
        # probar uses stateful metrics to determine which metric values to average.
        # Since this is only called on_epoch_end, no metrics should be averaged
        # i.e. all metrics should be considered 'stateful'.
        # don't set stateful_metrics in constructor because it wraps it in `set`.
        self.progbar.stateful_metrics = Universe()

    def on_epoch_end(self, epoch: int, logs=None):
        self.last_seen = epoch + 1
        self.progbar.update(epoch + 1, list(logs.items()))

    def on_train_end(self, logs=None):
        del logs
        if self.last_seen < self.progbar.target:
            if tf.version.VERSION < "2.3":
                sys.stdout.write("\n")
            else:
                self.progbar.update(self.last_seen, finalize=True)


@register
class EarlyStoppingV2(tf.keras.callbacks.EarlyStopping):
    """Same as EarlyStopping, but restores best weights even if not stopped early."""

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if (
            self.restore_best_weights
            and self.stopped_epoch == 0
            and self.best_weights is not None
        ):
            self.model.set_weights(self.best_weights)


@register
def get_model_callbacks(
    monitor: tp.Optional[str] = None,
    mode: tp.Optional[str] = None,
    patience: int = 10,
    terminate_on_nan: bool = False,
    log_dir: tp.Optional[str] = None,
) -> tp.List[tf.keras.callbacks.Callback]:
    """
    Convenience method for getting all callbacks used in most experiments.

    Args:
        monitor: if not None, adds an `EarlyStoppingV2`.
        mode: used in `EarlyStoppingV2`. If `None`, resets to 'auto'
        patience: used in `EarlyStoppingV2`.
        terminate_on_nan: if True, adds a `tf.keras.callbacks.TerminateOnNaN`.
        log_dir: if True, adds a `tf.keras.callbacks.TensorBoard`.

    Returns:
        List of up to three `tf.keras.callbacks.Callback`s.
    """
    out = []
    if monitor is not None:
        if mode is None:
            mode = "auto"
        out.append(
            EarlyStoppingV2(
                monitor=monitor, mode=mode, patience=patience, restore_best_weights=True
            )
        )
    if terminate_on_nan:
        out.append(tf.keras.callbacks.TerminateOnNaN())
    if log_dir is not None:
        out.append(
            tf.keras.callbacks.TensorBoard(
                log_dir, write_steps_per_second=True, profile_batch=(5, 10)
            )
        )
    return out
