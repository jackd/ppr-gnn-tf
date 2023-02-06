import typing as tp

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def sp_to_tf(matrix: sp.spmatrix, dtype=None) -> tf.SparseTensor:
    """Convert `sp.spmatrix` to `tf.SparseTensor`."""
    matrix: sp.coo_matrix = matrix.tocoo()
    matrix.eliminate_zeros()
    values = tf.convert_to_tensor(matrix.data, dtype)
    indices = tf.stack((matrix.row, matrix.col), axis=-1)
    indices = tf.cast(indices, tf.int64)
    return tf.SparseTensor(indices, values, matrix.shape)


def tf_to_sp(st: tf.SparseTensor) -> sp.spmatrix:
    return sp.coo_matrix(
        (st.values.numpy(), st.indices.numpy().T), shape=st.dense_shape.numpy()
    )


def sparse_normalize(
    matrix: sp.spmatrix,
    *,
    dtype: np.dtype = np.float32,
    symmetric: bool = True,
    renormalize: bool = False,
) -> sp.spmatrix:
    """
    Get a normalized `sp.spmatrix`.

    Args:
        matrix: input matrix.
        dtype: numpy dtype of returned matrix.
        symmetric: if True, returns D^{-1/2} A D^{-1/2}, otherwise D^{-1} A.
        renormalize: if True, first adds an identity matrix to `matrix`.

    Returns:
        normalized matrix. Rows where the row-sum is zero are filled with zeros.
    """
    if matrix.dtype != dtype:
        matrix = matrix.astype(dtype)

    if renormalize:
        matrix = matrix + sp.eye(matrix.shape[0], dtype=dtype)

    d = np.asarray(matrix.sum(1)).squeeze(1)
    matrix: sp.coo_matrix = matrix.tocoo()
    if symmetric:
        factor = 1 / np.sqrt(d)
        factor[d == 0] = 0
        data = matrix.data * factor[matrix.row] * factor[matrix.col]
        matrix = sp.coo_matrix((data, (matrix.row, matrix.col)), shape=matrix.shape)
    else:
        factor = 1 / d
        factor[d == 0] = 0
        data = matrix.data * factor[matrix.row]
        matrix = sp.coo_matrix((data, (matrix.row, matrix.col)), shape=matrix.shape)
    return matrix


def sparse_gather(matrix: sp.spmatrix, indices: np.ndarray) -> sp.spmatrix:
    """
    Get the result of slicing the specified rows from matrix.

    This is the sparse equivalent of `matrix[indices]`.

    Args:
        matrix: [N, M] sp.spmatrix
        indices: [P] int64, np.all(indices < N)

    Returns:
        [P, M] sp.spmatrix with the same dtype as `matrix`.
    """
    in_size = matrix.shape[0]
    (out_size,) = indices.shape
    mask = np.zeros((in_size,), dtype=bool)
    coo = matrix.tocoo()
    mask[indices] = True
    edge_mask = mask[coo.row]
    inv_indices = np.zeros((coo.shape[0],), dtype=np.int64)
    inv_indices[indices] = np.arange(out_size)
    return sp.coo_matrix(
        (coo.data[edge_mask], (inv_indices[coo.row[edge_mask]], coo.col[edge_mask])),
        shape=(out_size, *coo.shape[1:]),
    )


def get_largest_component_indices(
    adjacency: sp.spmatrix,
    *,
    directed: bool = True,
    connection="weak",
) -> tp.Optional[np.ndarray]:
    """
    Get the indices associated with the largest connected component.

    Args:
        adjacency: [n, n] adjacency matrix
        directed, connection: used in get_component_labels

    Returns:
        None if graph is connected, otherwise [size], int64 indices in [0, n) of nodes in the largest connected component,
            size <= n.
    """
    nc, labels = sp.csgraph.connected_components(
        adjacency, return_labels=True, directed=directed, connection=connection
    )
    if nc == 1:
        return None
    sizes = [np.count_nonzero(labels == i) for i in range(nc)]
    (indices,) = np.where(labels == np.argmax(sizes))
    return indices
