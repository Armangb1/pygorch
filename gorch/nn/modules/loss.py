from .module import Module
from gorch import Tensor
import gorch

class _Loss(Module):
    def __init__(self) -> None:
        super().__init__()

class MSELoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input:Tensor, target:Tensor):
        return gorch.mean((target-input)**2)