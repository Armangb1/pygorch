
from .optimizer import Optimizer
import numpy as np 
class Adadelta(Optimizer):
    
    def __init__(self, parameters, rho = 0.9, eps = 1e-8, lr = 1):
        super().__init__(parameters)

        self.rho = rho
        self.lr = lr
        self.eps = eps
        self.v = [np.zeros_like(p.value) for p in self.parameters]
        self.u = [np.zeros_like(p.value) for p in self.parameters]


    def step(self):
        
        lr = self.lr
        rho = self.rho
        eps = self.eps
        
        for i, param in enumerate(self.parameters):
            v = self.v[i]
            v = rho*v + (1-rho)*param.grad.value**2
            self.v[i] = v

            u = self.u[i]

            dw = np.sqrt(u+eps)/np.sqrt(v+eps) *param.grad.value

            u = rho*u + (1-rho)*dw**2
            self.u[i] = u
            param.value-= lr*dw
