from .optimizer import Optimizer
import numpy as np 

class RMSprop(Optimizer):
    
    def __init__(self, parameters, lr = 0.001, alpha = 0.99, eps = 1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.v = [np.zeros_like(p.value) for p in self.parameters]
        self.eps = eps

    def step(self):
        lr = self.lr
        alpha = self.alpha
        eps = self.eps
        for i, param in enumerate(self.parameters):

            v = self.v[i]

            v = alpha*v + (1-alpha)*param.grad.value**2

            self.v[i] = v

            param.value -= lr/(np.sqrt(v)+eps)*param.grad.value
