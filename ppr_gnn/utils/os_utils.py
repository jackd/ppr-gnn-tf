import functools
import os
import typing as tp

import gin

register = functools.partial(gin.register, module="ppr_gnn.utils.os_utils")


@register
def get_environ(key: str, default: str) -> str:
    """Configurable `os.environ.get`."""
    return os.environ.get(key, default)


@register
def get_dir(
    data_dir: tp.Optional[str], environ: str, default: tp.Optional[str]
) -> tp.Optional[str]:
    """Get an expanded path based on `data_dir` or environment variable `environ`."""
    if data_dir is None:
        data_dir = os.environ.get(environ, default)
    if data_dir is None:
        return None
    return os.path.expanduser(os.path.expandvars(data_dir))


@register
def join_star(args):
    """Expanded `os.path.join(*args)` registered with gin."""
    return os.path.expanduser(os.path.expandvars(os.path.join(*args)))
