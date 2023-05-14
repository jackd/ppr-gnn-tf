import functools
import time
import typing as tp

import gin
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tqdm

from ..data.types import DataSplit, PropTransitiveData
from . import callbacks as cb
from .layers import DropoutV2

register = functools.partial(gin.register, module="ppr_gnn.utils.train")


def _finalize(
    model: tf.keras.Model,
    validation_data: tp.Optional[tf.data.Dataset] = None,
    test_data: tp.Optional[tf.data.Dataset] = None,
) -> tp.Dict[str, float]:
    results = {}

    def evaluate(data: tf.data.Dataset):
        return model.evaluate(data, return_dict=True, verbose=len(data) > 1)

    if validation_data is not None:
        print("Evaluating validation_data...")
        val_res = evaluate(validation_data)
        results.update({f"val_{k}": v for k, v in val_res.items()})
        print("Evaluating test_data..")
    if test_data is not None:
        test_res = evaluate(test_data)
        results.update({f"test_{k}": v for k, v in test_res.items()})

    return results


def _get_callbacks(
    callbacks: tp.Iterable[tf.keras.callbacks.Callback],
    model: tf.keras.Model,
    verbose: int,
    steps_per_epoch: int,
    epochs: int,
) -> tf.keras.callbacks.CallbackList:
    is_single = steps_per_epoch == 1
    if is_single and verbose:
        callbacks = [*callbacks, cb.EpochProgbarLogger()]

    return tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=verbose != 0 and not is_single,
        model=model,
        verbose=verbose,
        epochs=epochs,
        steps=steps_per_epoch,
    )


@register
def benchmark_model_timings(
    split: DataSplit,
    model_fn: tp.Callable[[tp.Any], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    warmup_steps: int = 10,
    benchmark_steps: int = 100,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    print_fn: tp.Callable[[str], None] = print,
    show_progress: bool = True,
):
    train_data, validation_data, test_data = split

    input_spec = train_data.element_spec[0]
    # make leading dimensions None. This works around issues with metrics
    input_spec = tf.nest.map_structure(
        lambda spec: tf.TensorSpec(shape=(None, *spec.shape[1:]), dtype=spec.dtype),
        input_spec,
    )
    model: tf.keras.Model = model_fn(input_spec)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )

    def benchmark(fn: tp.Callable, desc: str):
        it = (
            tqdm.trange(warmup_steps, desc=f"Warming up {desc}...")
            if show_progress
            else range(warmup_steps)
        )
        for _ in it:
            fn()

        t0 = time.time()
        it = (
            tqdm.trange(benchmark_steps, desc=f"Benchmarking {desc}...")
            if show_progress
            else range(benchmark_steps)
        )
        for _ in it:
            fn()
        t1 = time.time()
        dt = t1 - t0
        print_fn(
            f"Completed {benchmark_steps} {desc}s in {int(np.round(dt*1e6))}μs, "
            f"{int(np.round(dt / benchmark_steps * 1e6))}μs / step"
        )
        return dt / benchmark_steps

    train_step = model.make_train_function()
    train_iter = iter(train_data.repeat().prefetch(prefetch_buffer))
    benchmark(lambda: train_step(train_iter), "train step")

    test_step = model.make_test_function()
    predict_step = model.make_predict_function()
    if validation_data is not None:
        validation_iter = iter(validation_data.repeat().prefetch(prefetch_buffer))
        benchmark(lambda: test_step(validation_iter), "validation_data test step")
        benchmark(lambda: predict_step(validation_iter), "validation_data predict step")

    if test_data is not None:
        test_iter = iter(test_data.repeat().prefetch(prefetch_buffer))
        benchmark(lambda: test_step(test_iter), "test_data test step")
        benchmark(lambda: predict_step(test_iter), "test_data predict step")


@register
def benchmark_memory_usage(
    split: DataSplit,
    model_fn: tp.Callable[[tp.Any], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    warmup_steps: int = 10,
    benchmark_steps: int = 100,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    print_fn: tp.Callable[[str], None] = print,
    show_progress: bool = True,
):
    train_data, validation_data, test_data = split

    input_spec = train_data.element_spec[0]
    # make leading dimensions None. This works around issues with metrics
    input_spec = tf.nest.map_structure(
        lambda spec: tf.TensorSpec(shape=(None, *spec.shape[1:]), dtype=spec.dtype),
        input_spec,
    )
    model: tf.keras.Model = model_fn(input_spec)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )

    def benchmark(fn: tp.Callable, desc: str):
        it = (
            tqdm.trange(warmup_steps, desc=f"Warming up {desc}...")
            if show_progress
            else range(warmup_steps)
        )
        for _ in it:
            fn()

        peaks = []
        it = (
            tqdm.trange(warmup_steps, desc=f"Benchmarking {desc}...")
            if show_progress
            else range(benchmark_steps)
        )
        for _ in it:
            tf.config.experimental.reset_memory_stats("GPU:0")
            fn()
            stats = tf.config.experimental.get_memory_info("GPU:0")
            peaks.append(stats["peak"])
        peaks = np.array(peaks) / 1024**2  # M
        print_fn(
            f"Completed {benchmark_steps} {desc}s\n"
            f"{np.mean(peaks)}  ± {np.std(peaks)}M\n"
            f"min: {np.min(peaks)}M, \n"
            f"max: {np.max(peaks)}M"
        )
        return peaks

    step = model.make_train_function()
    data_iter = iter(train_data.repeat().prefetch(prefetch_buffer))
    benchmark(lambda: step(data_iter), "train step")

    step = model.make_test_function()
    if validation_data is not None:
        data_iter = iter(validation_data.repeat().prefetch(prefetch_buffer))
        benchmark(lambda: step(data_iter), "validation_step")
    if test_data is not None:
        data_iter = iter(test_data.repeat().prefetch(prefetch_buffer))
        benchmark(lambda: step(data_iter), "test_step")


@register
def build_fit_test(
    split: DataSplit,
    model_fn: tp.Callable[[tp.Any], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    callbacks: tp.Iterable[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_freq: int = 1,
    verbose: bool = True,
    steps_per_epoch: tp.Optional[int] = None,
    prefetch_buffer: tp.Optional[int] = tf.data.AUTOTUNE,
) -> tp.Tuple[tf.keras.Model, tf.keras.callbacks.History, tp.Dict[str, tp.Any]]:
    """
    Build a model, fit it to data and evaluate.

    Args:
        split: DataSplit. Each entry must have compatible `element_spec`.
        model_fn: function mapping input_sec -> tf.keras.Model.
        optimizer: used in Model.compile.
        loss: used in Model.compile.
        metrics: used in Model.compile.
        weighted_metrics: used in Model.compile.
        callbacks, epochs, validation_freq, verbose, steps_per_epch: see `Model.fit`
        prefetch_buffer: if not None, used to `prefetch` each dataset in `split`.

    Returns tuple containing:
        trained tf.keras.Model
        History object, result of Model.fit
        results: dict mapping f'{prefix}_{metric_name}` -> metric_val, where `prefix`
            is one of 'val', 'test'.
    """
    train_data, validation_data, test_data = split
    del split

    if prefetch_buffer is not None:
        train_data, validation_data, test_data = (
            d if d is None else d.prefetch(prefetch_buffer)
            for d in (train_data, validation_data, test_data)
        )

    input_spec = train_data.element_spec[0]
    # make leading dimensions None. This works around issues with metrics
    input_spec = tf.nest.map_structure(
        lambda spec: tf.TensorSpec(shape=(None, *spec.shape[1:]), dtype=spec.dtype),
        input_spec,
    )
    model: tf.keras.Model = model_fn(input_spec)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )
    if verbose:
        model.summary()

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    callbacks = _get_callbacks(
        callbacks,
        model=model,
        steps_per_epoch=steps_per_epoch,
        verbose=verbose,
        epochs=epochs,
    )
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        validation_freq=validation_freq,
        callbacks=callbacks,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
    )
    results = _finalize(model, validation_data, test_data)
    return model, history, results


def _v2_split(model: tf.keras.Model) -> tp.Tuple[tf.keras.layers.Layer, tf.keras.Model]:
    layer = None
    for layer in model.layers:
        if isinstance(
            layer, (tf.keras.layers.Dropout, tf.keras.layers.InputLayer, DropoutV2)
        ):
            continue
        if isinstance(layer, tf.keras.layers.Dense):
            break
        raise Exception(f"Unrecognized layer {layer}")
    else:
        raise Exception("No dense layer found")
    assert layer is not None
    model = tf.keras.Model(layer.output, model.output)
    return layer, model


@register
def build_fit_test_propagate_mlp_v2(
    data: PropTransitiveData,
    model_fn: tp.Callable[[tf.TensorSpec], tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metrics=None,
    weighted_metrics=None,
    batch_size: int = -1,
    callbacks: tp.Iterable[tf.keras.callbacks.Callback] = (),
    epochs: int = 1,
    validation_freq: int = 1,
    verbose: bool = True,
    steps_per_epoch: tp.Optional[int] = None,
    prefetch_buffer: tp.Optional[int] = tf.data.AUTOTUNE,
    skip_validation_during_training: bool = False,
) -> tp.Tuple[tf.keras.Model, tf.keras.callbacks.History, tp.Dict[str, tp.Any]]:
    """
    Similar to `build_fit_test` but for propagate-mlp models on large datasets.

    Applicable when `data` has a small number of training labels but a large number of
    both input features and test labels.

    Propagation is done on cpu, so uses numpy.

    For training, we precompute `(M @ propagator) @ node_features`.

    For evaluation on validation/test sets, we compute
    `M @ (propagator @ (node_features @ kernel))`, where `kernel` is the first dense
    layer's kernel.

    Args:
        data: source `PropTransitiveData`
        propagator_fn: numpy function mapping sp.spmatrix -> (np.ndarray -> np.ndarray)
        model_fn: function mapping `tf.TensorSpec` to a `tf.keras.Model`. The returned
            model must have a dense layer as the first layer that isn't either an input
            or dropout layer.
        optimizer, loss, metrics, weighted_metrics: used in `tf.keras.Model.compile`
        batch_size: batch size used during training/evaluation/testing. If -1, uses a
            single batch per dataset.
        callbacks, epochs, verbose, steps_per_epoch: see `tf.keras.Model.fit`
        prefetch_buffer: if not None, applied with `tf.data.Dataset.prefetch` to each
            dataset.
        skip_validation_during_training: if True, `validation_data is None` in
            `tf.keras.Model.fit` call.

    Returns tuple containing:
        trained tf.keras.Model
        History object, result of Model.fit
        results: dict mapping f'{prefix}_{metric_name}` -> metric_val, where `prefix`
            is one of 'val', 'test'.
    """
    node_features: sp.spmatrix = data.node_features
    propagator = data.propagator
    node_features_T = node_features.T
    num_nodes, num_features = data.node_features.shape

    def get_dataset(
        ids: tp.Optional[np.ndarray], training: bool
    ) -> tp.Optional[tf.data.Dataset]:
        if ids is None:
            return None
        (num_labels,) = ids.shape
        features = np.empty((num_labels, num_features), dtype=node_features_T.dtype)
        labels = data.labels[ids]
        if verbose:
            ids = tqdm.tqdm(
                ids, desc=f"Creating {'train' if training else 'validation'} mlp data"
            )
        rhs = np.zeros((num_nodes,))
        for i, id_ in enumerate(ids):
            rhs[id_] = 1
            propagated_features = propagator(rhs)  # [num_nodes]
            features[i] = node_features_T @ propagated_features
            rhs[id_] = 0

        example = tf.convert_to_tensor(features), tf.convert_to_tensor(labels)
        if batch_size == -1 or batch_size >= num_labels:
            ds = tf.data.Dataset.from_tensors(example)
        else:
            ds = tf.data.Dataset.from_tensor_slices(example)
            if training:
                ds = ds.shuffle(num_labels)
            ds = ds.batch(batch_size)
        if prefetch_buffer is not None:
            ds = ds.prefetch(prefetch_buffer)
        return ds

    train_data = get_dataset(data.train_ids, training=True)
    validation_data = (
        None
        if skip_validation_during_training
        else get_dataset(data.validation_ids, training=False)
    )

    spec = tf.TensorSpec((None, num_features), dtype=tf.float32)
    model: tf.keras.Model = model_fn(spec)
    compile_kwargs = dict(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )
    model.compile(**compile_kwargs)
    if verbose:
        model.summary()

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    callbacks = _get_callbacks(
        callbacks,
        model=model,
        steps_per_epoch=steps_per_epoch,
        verbose=verbose,
        epochs=epochs,
    )
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        validation_freq=validation_freq,
        callbacks=callbacks,
        verbose=verbose,
        steps_per_epoch=steps_per_epoch,
    )
    layer, rest_model = _v2_split(model)
    rest_model.compile(**compile_kwargs)
    if verbose:
        print("Computing node_features @ kernel")
    features = node_features @ layer.kernel.numpy()  # [num_nodes, units]
    if layer.use_bias:
        features += layer.bias.numpy()
    units = features.shape[1]
    if verbose:
        cols = tqdm.trange(features.shape[1], desc="Propagating test features")
    else:
        cols = range(units)
    propagated = np.empty((num_nodes, units))
    for i in cols:
        propagated[:, i] = propagator(features[:, i])

    def get_dataset_for_test(
        ids: tp.Optional[np.ndarray],
    ) -> tp.Optional[tf.data.Dataset]:
        if ids is None:
            return None
        features = propagated[ids]
        labels = data.labels[ids]
        example = tf.convert_to_tensor(features), tf.convert_to_tensor(labels)
        if batch_size == -1 or batch_size >= ids.shape[0]:
            ds = tf.data.Dataset.from_tensors(example)
        else:
            ds = tf.data.Dataset.from_tensor_slices(example).batch(batch_size)
        if prefetch_buffer:
            ds = ds.prefetch(prefetch_buffer)
        return ds

    results = _finalize(
        rest_model,
        get_dataset_for_test(data.validation_ids),
        get_dataset_for_test(data.test_ids),
    )
    return model, history, results
