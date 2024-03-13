import torch

from ency.loader.lazy_dep_configs import lazy_dep_configs
from ency.loader.lazy_dep_loader import LazyDependencyLoader

libflash_attn_ops = LazyDependencyLoader(lazy_dep_configs["libflash_attn_ops"])
torch.ops.load_library(libflash_attn_ops.get())

liblinalg_ops = LazyDependencyLoader(lazy_dep_configs["liblinalg_ops"])
torch.ops.load_library(liblinalg_ops.get())