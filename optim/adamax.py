from .optimizer import Optimizer
import numpy as np

class Adamax(Optimizer):
    def __init__(self, parameters, lr=0.001, betas = (0.9, 0.999), eps = 1e-8):
        super().__init__(parameters)
        self.m = [np.zeros_like(p.value) for p in self.parameters]
        self.u = [np.zeros_like(p.value) for p in self.parameters]
        self.lr = lr
        self.betas = betas
        self.eps = eps


    def step(self):
        
        lr = self.lr 
        betas = self.betas 
        eps = self.eps 

        beta1 = betas[0]
        beta2 = betas[1]

        for i, param in enumerate(self.parameters):
            m = self.m[i]
            u = self.u[i]
            m = beta1*m + (1-beta1)*param.grad.value
            self.m[i] = m

            u = np.maximum(beta2*u,np.abs(param.grad.value)+eps)
            self.u[i] = u

            m = m/(1-beta1)

            param -= lr*m/u
