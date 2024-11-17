import numpy as np
from ..tensor import Tensor

class Parameter(Tensor):
    def __init__(self, shape) -> None:
        value = np.random.randn(*shape)
        super().__init__(value, requires_grad=True)

    def __repr__(self) -> str:
        return "Parameter containing:" + "\n" +super().__repr__()
