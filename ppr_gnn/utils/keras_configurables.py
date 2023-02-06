"""Importing this registers various parts of `tf.keras` with gin."""
import gin
import tensorflow as tf

for opt in (
    tf.keras.optimizers.Adadelta,
    tf.keras.optimizers.Adagrad,
    tf.keras.optimizers.Adam,
    tf.keras.optimizers.Adamax,
    tf.keras.optimizers.Ftrl,
    tf.keras.optimizers.Nadam,
    tf.keras.optimizers.RMSprop,
    tf.keras.optimizers.SGD,
):
    gin.register(opt, module="tf.keras.optimizers")

for reg in (
    tf.keras.regularizers.L1,
    tf.keras.regularizers.L1L2,
    tf.keras.regularizers.L2,
):
    gin.register(reg, module="tf.keras.regularizers")

for cb in (
    tf.keras.callbacks.CSVLogger,
    tf.keras.callbacks.EarlyStopping,
    tf.keras.callbacks.History,
    tf.keras.callbacks.LambdaCallback,
    tf.keras.callbacks.LearningRateScheduler,
    tf.keras.callbacks.ModelCheckpoint,
    tf.keras.callbacks.ProgbarLogger,
    tf.keras.callbacks.ReduceLROnPlateau,
    tf.keras.callbacks.RemoteMonitor,
    tf.keras.callbacks.TensorBoard,
    tf.keras.callbacks.TerminateOnNaN,
):
    gin.register(cb, module="tf.keras.callbacks")

for loss in (
    tf.keras.losses.BinaryCrossentropy,
    tf.keras.losses.CategoricalCrossentropy,
    tf.keras.losses.CategoricalHinge,
    tf.keras.losses.CosineSimilarity,
    tf.keras.losses.Hinge,
    tf.keras.losses.Huber,
    tf.keras.losses.KLD,
    tf.keras.losses.KLDivergence,
    tf.keras.losses.LogCosh,
    tf.keras.losses.MAE,
    tf.keras.losses.MAPE,
    tf.keras.losses.MSE,
    tf.keras.losses.MSLE,
    tf.keras.losses.MeanAbsoluteError,
    tf.keras.losses.MeanAbsolutePercentageError,
    tf.keras.losses.MeanSquaredError,
    tf.keras.losses.MeanSquaredLogarithmicError,
    tf.keras.losses.Poisson,
    tf.keras.losses.SparseCategoricalCrossentropy,
    tf.keras.losses.SquaredHinge,
):
    gin.register(loss, module="tf.keras.losses")

for metric in (
    tf.keras.metrics.AUC,
    tf.keras.metrics.Accuracy,
    tf.keras.metrics.BinaryAccuracy,
    tf.keras.metrics.BinaryCrossentropy,
    tf.keras.metrics.CategoricalAccuracy,
    tf.keras.metrics.CategoricalCrossentropy,
    tf.keras.metrics.CategoricalHinge,
    tf.keras.metrics.CosineSimilarity,
    tf.keras.metrics.FalseNegatives,
    tf.keras.metrics.FalsePositives,
    tf.keras.metrics.Hinge,
    tf.keras.metrics.KLD,
    tf.keras.metrics.KLDivergence,
    tf.keras.metrics.LogCoshError,
    tf.keras.metrics.MAE,
    tf.keras.metrics.MAPE,
    tf.keras.metrics.MSE,
    tf.keras.metrics.MSLE,
    tf.keras.metrics.Mean,
    tf.keras.metrics.MeanAbsoluteError,
    tf.keras.metrics.MeanAbsolutePercentageError,
    tf.keras.metrics.MeanIoU,
    tf.keras.metrics.MeanMetricWrapper,
    tf.keras.metrics.MeanRelativeError,
    tf.keras.metrics.MeanSquaredError,
    tf.keras.metrics.MeanSquaredLogarithmicError,
    tf.keras.metrics.MeanTensor,
    tf.keras.metrics.Poisson,
    tf.keras.metrics.Precision,
    tf.keras.metrics.PrecisionAtRecall,
    tf.keras.metrics.Recall,
    tf.keras.metrics.RecallAtPrecision,
    tf.keras.metrics.RootMeanSquaredError,
    tf.keras.metrics.SensitivityAtSpecificity,
    tf.keras.metrics.SparseCategoricalAccuracy,
    tf.keras.metrics.SparseCategoricalCrossentropy,
    tf.keras.metrics.SparseTopKCategoricalAccuracy,
    tf.keras.metrics.SpecificityAtSensitivity,
    tf.keras.metrics.SquaredHinge,
    tf.keras.metrics.Sum,
    tf.keras.metrics.TopKCategoricalAccuracy,
    tf.keras.metrics.TrueNegatives,
    tf.keras.metrics.TruePositives,
):
    gin.register(metric, module="tf.keras.metrics")
