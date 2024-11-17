from .module import Module
import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn.parameter import Parameter

class Linear(Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.weight = Parameter((in_dim, out_dim))
        self.bias = Parameter((1, out_dim))
        self.add_parameter('Weight', self.weight)
        self.add_parameter('bias', self.bias)

    def forward(self, inputs, **kwargs):
        out = inputs@self.weight+self.bias
        return out
