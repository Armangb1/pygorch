from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameter, lr = 0.001):
        super().__init__(parameter)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.value -= param.grad.value*self.lr
