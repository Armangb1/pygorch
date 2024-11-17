from .optimizer import Optimizer
import numpy as np
# this is a basic implementation of
# Adagrad. this optimizer can be improved with
# weigth decay rate an lr decay
class Adagrad(Optimizer):
    def __init__(self, parameters, lr=0.03, eps = 1e-7):
        super().__init__(parameters)
        self.lr = lr
        self.S = [np.zeros_like(p.value) for p in self.parameters]
        self.eps = eps

    def step(self):
        eps = self.eps
        lr = self.lr
        for i, param in enumerate(self.parameters):
            S = self.S[i]
            S = S + param.grad.value**2
            self.S[i] = S
            param.value -= lr/(np.sqrt(S)+eps) *param.grad.value
        

    