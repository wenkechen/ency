import torch
from torch.autograd.function import BackwardCFunction

from ency.utils import FastFunction


class ActivationFunction(FastFunction):
    @staticmethod
    def fast_forward(x: torch.Tensor, activation_type: int = 1):
        return torch.ops.linalg.activation_fwd(x, activation_type)

    @staticmethod
    def forward(ctx: BackwardCFunction, x: torch.Tensor, activation_type: int = 1):
        output = torch.ops.linalg.activation_fwd(x, activation_type)
        ctx.save_for_backward(x)
        ctx.activation_type = activation_type
        return output

    @staticmethod
    def backward(ctx: BackwardCFunction, do: torch.Tensor):
        x = ctx.saved_tensors[0]
        dx = torch.ops.linalg.activation_bwd(do, x, ctx.activation_type)
        return dx


class FastActivation(torch.nn.Module):
    def __init__(self, activation_type: int = 1) -> None:
        super().__init__()
        assert activation_type in [0, 1, 2, 3]
        self.activation_type = activation_type

    def forward(self, x):
        return ActivationFunction.apply(x, self.activation_type)
