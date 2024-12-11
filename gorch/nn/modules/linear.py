from .module import Module
from ..parameter import Parameter
import gorch
class Linear(Module):
    def __init__(self, in_dim, out_dim, bias:bool = True) -> None:
        super().__init__()
        self.weight = Parameter((in_dim, out_dim))
        self.add_parameter('Weight', self.weight)
        if bias:
            self.bias = Parameter((1, out_dim))
            self.add_parameter('bias', self.bias)
        else:
            self.bias = gorch.Tensor([[0]])

    def forward(self, inputs, **kwargs):
        out = inputs@self.weight+self.bias
        return out
