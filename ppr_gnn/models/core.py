import functools

import gin

register = functools.partial(gin.register, module="ppr_gnn.models")
