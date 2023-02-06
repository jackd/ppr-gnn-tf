import functools
import typing as tp

import gin
import numpy as np
import scipy.sparse as sp

from ..utils.scipy_utils import sparse_normalize

register = functools.partial(gin.register, module="ppr_gnn.data.transforms")


@register
def row_normalize(x: tp.Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    """Row-normalize `x` by dividing by it's sum."""
    if sp.issparse(x):
        return sparse_normalize(x, symmetric=False)
    factor = np.asarray(x).sum(1, keepdims=True)
    factor[factor == 0] = 1
    out = x / factor
    return out


register(sparse_normalize)
