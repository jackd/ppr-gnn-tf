"""
Generic entry-point for programs predominantly configured via `gin-config`.

Parses arbitrary config files and bindings. It is assumed that one of these
configures a `main.fun`.

Positional args are interpreted as paths to config files, with `.gin` appended
if missing.

Usage:
```python
python -m ppr_gnn train/build-fit-test.gin mlp-ppr/dagnn/cora.gin --bindings="
  train_preprocess = False
  log_dir = '/tmp/ppr-gnn/tensorboard-logs/mlp-ppr/dagnn/cora'
"
```

General notes:
    - bindings may be repeated or separated by new-lines
    - `ppr_gnn/config` directory is automatically added to gin search path
"""

import os

import gacl
import gin
import tensorflow as tf  # stops weird bugs - maybe dgl interop?

if __name__ == "__main__":
    gin.config.add_config_file_search_path(
        os.path.join(os.path.dirname(__file__), "config")
    )
    gacl.cli_main()
