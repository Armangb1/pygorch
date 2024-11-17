from abc import ABC
from ..tensor.tensor import Tensor

class Optimizer(ABC):
    def __init__(self, parameters):
        if isinstance(parameters, Tensor):
            raise TypeError("parameters should be an iterable but got {}".format(type(parameters)))
        elif isinstance(parameters, dict):
            parameters.values()
        self.parameters = list(parameters)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None