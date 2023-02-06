import typing as tp

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


class DataSplit(tp.NamedTuple):
    train_data: tf.data.Dataset
    validation_data: tp.Optional[tf.data.Dataset]
    test_data: tp.Optional[tf.data.Dataset]


def _replace(original, replacement):
    return original if replacement is None else replacement


class TransitiveData(tp.NamedTuple):
    """Data for a node classification transitive graph problem."""

    node_features: tp.Union[np.ndarray, sp.spmatrix]
    adjacency: sp.spmatrix
    labels: np.ndarray
    train_ids: tp.Optional[np.ndarray]
    validation_ids: tp.Optional[np.ndarray]
    test_ids: tp.Optional[np.ndarray]

    def rebuild(
        self,
        *,
        node_features: tp.Optional[tp.Union[np.ndarray, sp.spmatrix]] = None,
        adjacency: tp.Optional[sp.spmatrix] = None,
        labels: tp.Optional[np.ndarray] = None,
        train_ids: tp.Optional[tp.Optional[np.ndarray]] = None,
        validation_ids: tp.Optional[tp.Optional[np.ndarray]] = None,
        test_ids: tp.Optional[tp.Optional[np.ndarray]] = None,
    ) -> "TransitiveData":
        return TransitiveData(
            node_features=_replace(self.node_features, node_features),
            adjacency=_replace(self.adjacency, adjacency),
            labels=_replace(self.labels, labels),
            train_ids=_replace(self.train_ids, train_ids),
            validation_ids=_replace(self.validation_ids, validation_ids),
            test_ids=_replace(self.test_ids, test_ids),
        )


class MLPData(tp.NamedTuple):
    """Data required for simple MLP training."""

    features: np.ndarray
    labels: np.ndarray


class PropTransitiveData(tp.NamedTuple):
    """Same as `TransitiveData` but with propagator instead of adjacency"""

    node_features: tp.Union[np.ndarray, sp.spmatrix]
    propagator: tp.Callable[[np.ndarray], np.ndarray]
    labels: np.ndarray
    train_ids: tp.Optional[np.ndarray]
    validation_ids: tp.Optional[np.ndarray]
    test_ids: tp.Optional[np.ndarray]

    def rebuild(
        self,
        *,
        node_features: tp.Optional[tp.Union[np.ndarray, sp.spmatrix]] = None,
        propagator: tp.Optional[tp.Callable[[np.ndarray], np.ndarray]] = None,
        labels: tp.Optional[np.ndarray] = None,
        train_ids: tp.Optional[tp.Optional[np.ndarray]] = None,
        validation_ids: tp.Optional[tp.Optional[np.ndarray]] = None,
        test_ids: tp.Optional[tp.Optional[np.ndarray]] = None,
    ) -> "PropTransitiveData":
        return PropTransitiveData(
            node_features=_replace(self.node_features, node_features),
            propagator=_replace(self.propagator, propagator),
            labels=_replace(self.labels, labels),
            train_ids=_replace(self.train_ids, train_ids),
            validation_ids=_replace(self.validation_ids, validation_ids),
            test_ids=_replace(self.test_ids, test_ids),
        )
