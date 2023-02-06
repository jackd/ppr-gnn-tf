import functools
import os
import shutil
import zipfile

import gin
import numpy as np
import pandas as pd
import scipy.sparse as sp
import wget

from .types import TransitiveData

register = functools.partial(gin.register, module="ppr_gnn.data.papers100m")


def _load_adj(
    row: np.ndarray, col: np.ndarray, n: int, make_symmetric: bool
) -> sp.coo_matrix:
    if make_symmetric:
        print("Making symmetric...")
        valid = row != col
        row = row[valid]
        col = col[valid]

        row, col = np.concatenate((row, col)), np.concatenate((col, row))
        i1d: np.ndarray = np.ravel_multi_index((row, col), (n, n))
        i1d.sort()  # pylint: disable=no-member
        row, col = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
            i1d, (n, n)
        )
        print("Finished making symmetric")

    return sp.coo_matrix(
        (np.ones(row.shape, dtype=np.float32), (row, col)), shape=(n, n)
    )


@register
def get_papers100m_data(ogb_dir: str, *, make_symmetric: bool = True) -> TransitiveData:
    """
    Get ogbn-papers100m dataset.

    Data is potentially downloaded/extracted if not already there. Note it is very large
    and may not fit in memory.
        - 111_059_956 nodes, 128 features per node
        - 1_615_685_872 edges
        - 1_207_179 training labels
        - 125_265 validation labels
        - 214_338 test labels

    Args:
        ogb_dir: directory to store ogb data.
        make_symmetric: if True, the returned adjacency matrix is symmetric.

    Returns:
        TransitiveData
    """
    data_dir = os.path.join(ogb_dir, "ogbn_papers100M")

    url = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
    split_paths = tuple(
        os.path.join(data_dir, "split", "time", f"{split}.csv.gz")
        for split in ("train", "valid", "test")
    )

    def has_raw_data():
        raw_paths = (
            (os.path.join(data_dir, "raw", fn)) for fn in ("data.npz", "node-label.npz")
        )
        has_raw_data = all(os.path.exists(rp) for rp in raw_paths)
        has_split_data = all(os.path.exists(sp) for sp in split_paths)
        return has_raw_data and has_split_data

    if not has_raw_data():
        zip_path = os.path.join(ogb_dir, "papers100M-bin.zip")
        if not os.path.exists(zip_path):
            # download
            print("Downloading papers100M data...")
            wget.download(url, zip_path)
        assert os.path.exists(zip_path)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(ogb_dir)
        shutil.move(
            os.path.join(ogb_dir, "papers100M-bin"),
            data_dir,
        )
        assert has_raw_data()
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("Finished extraction.")

    # copy split indices
    splits = {}
    for split, path in zip(("train", "validation", "test"), split_paths):
        splits[split] = (
            pd.read_csv(path, compression="gzip", header=None)
            .values.T[0]
            .astype(np.int64)
        )
    labels = np.load(os.path.join(data_dir, "raw", "node-label.npz"))
    labels = labels["node_label"]
    labels[np.isnan(labels)] = -1
    labels = labels.astype(np.int64)
    if labels.ndim == 2:
        labels = np.squeeze(labels, axis=1)

    data = np.load(os.path.join(data_dir, "raw", "data.npz"))
    print("Loading node features...")
    node_features = data["node_feat"]
    print("Loading adjacency...")
    row, col = data["edge_index"]
    adj = _load_adj(row, col, node_features.shape[0], make_symmetric=make_symmetric)
    print("Got ogbn-papers100m TransitiveData")
    return TransitiveData(
        node_features,
        adj,
        labels,
        splits["train"],
        splits["validation"],
        splits["test"],
    )
