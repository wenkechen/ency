import logging
import os
import subprocess
import sys
import urllib.request
from itertools import takewhile
from typing import Callable, List, Literal, Optional, Union

import tqdm
from filelock import FileLock
from pydantic import BaseModel, BaseSettings, root_validator

logger = logging.getLogger(__name__)


class LoaderEnv(BaseModel):
    python_version: List[int]
    pytorch_version: List[int]
    pytorch_cuda_version: List[int]


StrOrTemplateFunc = Union[str, Callable[[LoaderEnv], str]]


class LazyLoaderConfig(BaseSettings):
    dep_type: Literal["file"]

    # for files
    local_path: Optional[str]
    is_executable: Optional[bool]

    # for pip
    package_name: Optional[str]
    whl_path: Optional[StrOrTemplateFunc]
    install_command: Optional[StrOrTemplateFunc]

    @root_validator
    def check_type_argumennts(cls, values):
        if values.get("dep_type") == "file":
            if any([values.get(k) is not None for k in ["local_path"]]):
                raise ValueError("local_path should exists for files")
        return values


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def parse_version(s: str):
    return tuple([int("".join(takewhile(str.isdigit, seg))) for seg in s.split(".")])


def get_env():
    torch_version = ()
    torch_cuda_version = ()
    try:
        import torch

        torch_version = parse_version(torch.__version__)
        torch_cuda_version = parse_version(torch.version.cuda)
    except:
        pass

    return LoaderEnv(
        python_version=(sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
        pytorch_version=torch_version,
        pytorch_cuda_version=torch_cuda_version,
    )


class LazyDependencyLoader:
    def __init__(self, config: LazyLoaderConfig) -> None:
        self.config = config

    def get(self):
        if self.config.dep_type == "file":
            return self.get_file()
        else:
            raise ValueError(f"dep type {self.config.dep_type} not supported")

    def get_file(self):
        os.makedirs(os.path.dirname(self.config.local_path), exist_ok=True)
        with FileLock(f"{self.config.local_path}.lock"):
            if not os.path.exists(self.config.local_path):
                raise RuntimeError(f"{self.config.local_path} does not exist")
            return self.config.local_path
