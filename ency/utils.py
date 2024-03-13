from __future__ import annotations

from typing import Any

import torch


class FastFunction(torch.autograd.Function):
    @staticmethod
    def fast_forward(*args, **kwargs):
        raise NotImplementedError("fast_forward not implementd for this function")

    @classmethod
    def apply(cls, *args, **kwargs):
        if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            return super().apply(*args, **kwargs)
        return cls.fast_forward(*args, **kwargs)
