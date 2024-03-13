import os

import ency
from ency.loader.lazy_dep_loader import LazyLoaderConfig, get_env

root = os.path.dirname(ency.__file__)

pytorch_version = ".".join(map(str, get_env().pytorch_version[:2]))
cuda_version = ".".join(map(str, get_env().pytorch_cuda_version))

lazy_dep_configs = {
    "libflash_attn_ops": LazyLoaderConfig(
        dep_type="file",
        local_path=os.path.join(root, "ops/lib/libflash_attn_ops.so"),
    ),
    "liblinalg_ops": LazyLoaderConfig(
        dep_type="file",
        local_path=os.path.join(root, "ops/lib/liblinalg_ops.so"),
    ),
}
