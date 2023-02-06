import functools
import typing as tp

import gin
import numpy as np
import tensorflow as tf
from gacl.callbacks import Callback, GinConfigLogger

register = functools.partial(gin.register, module="ppr_gnn.utils.experiment_callbacks")


def print_result_stats(
    results: tp.Union[tp.Mapping, tp.Sequence[tp.Mapping]],
    print_fn: tp.Callable[[str], None] = print,
):
    """Merge a list of maps and print the mean ± std."""
    if hasattr(results, "items"):
        merged = results
    else:
        merged = {}
        for r in results:
            for k, v in r.items():
                merged.setdefault(k, []).append(v)

    print_results(
        {
            k: f"{np.mean(v)} ± {np.std(v)}"
            for k, v in merged.items()
            if not isinstance(v, str)
        },
        print_fn=print_fn,
    )


def print_results(
    results: tp.Mapping[str, tp.Any], print_fn: tp.Callable[[str], None] = print
):
    """Print a map with values aligned."""
    if not results:
        return
    width = max(len(k) for k in results) + 1
    for k in sorted(results):
        print_fn(f"{k.ljust(width)}: {results[k]}")


@register
class FitReporter(Callback):
    """gacl `Callback` that prints results per experiment and stats at end."""

    def __init__(self):
        self.metrics = []

    def on_trial_completed(self, trial_id: int, result):
        if hasattr(result, "items"):
            metrics = result
        else:
            model, history, metrics = result
            del model, history
        print(f"Completed trial {trial_id}")
        print_results(metrics)
        self.metrics.append(metrics)

    def on_end(self):
        if len(self.metrics) > 1:
            print(f"Completed {len(self.metrics)} trials")
            print_result_stats(self.metrics)


@register
class KerasSeedSetter(Callback):
    """gacl `Callback` that calls `keras.utils.set_random_seed` at start of each run."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.seeds = None

    def on_start(self, num_trials: tp.Optional[int]):
        assert num_trials is not None
        rng = np.random.default_rng(self.seed)
        self.seeds = rng.integers(0, high=np.iinfo(np.int32).max, size=num_trials)

    def on_trial_start(self, trial_id: int):
        tf.keras.utils.set_random_seed(int(self.seeds[trial_id]))


@register
class EnableOpDeterminism(Callback):
    """gacl `Callback` that calls `tf.config.experimental.enable_op_determinism`."""

    def on_start(self, num_trials: tp.Optional[int]):
        del num_trials
        tf.config.experimental.enable_op_determinism()


@register
def get_experiment_callbacks(
    reporter: tp.Optional[Callback] = None,
    seed_setter: tp.Optional[Callback] = None,
    enable_op_determinism: bool = True,
    log_config: bool = True,
) -> tp.List[Callback]:
    """
    Convenience method that returns commonly used gacl `Callback`s.

    Args:
        reporter: some `Callback` that can be used to report results, e.g. `FitReporter`
        seed_setter: some `Callback` that can be used to set seeds, e.g.
            `KerasSeedSetter`
        enable_op_determinism: if True, returned list constains a `EnableOpDeterminism`
        log_config: if True, returned list contains a `gacl.callbacks.GinConfigLogger`

    Returns:
        List of `gacl.callbacks.Callback`.
    """
    callbacks = []
    if enable_op_determinism:
        callbacks.append(EnableOpDeterminism())
    if seed_setter:
        callbacks.append(seed_setter)
    if reporter:
        callbacks.append(reporter)
    if log_config:
        callbacks.append(GinConfigLogger())
    return callbacks
