from .optimizer import Optimizer
import numpy as np

class SGDM(Optimizer):
    def __init__(self, parameters, lr = 0.001, beta = 0.9):
        super().__init__(parameters)
        self.lr = lr
        self.m = [np.zeros_like(p.value) for p in self.parameters]
        self.beta = beta


    def step(self):
        beta = self.beta
        for i, param in enumerate(self.parameters):
            m = self.m[i]
            m = beta * m + (1-beta)*param.grad.value
            param.value -= self.lr*m
            self.m[i] = m


             
        
        