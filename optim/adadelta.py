
from .optimizer import Optimizer
import numpy as np 
class Adadelta(Optimizer):
    
    def __init__(self, rho = 0.9, eps = 1e-8, lr = 1):
        super().__init__()

        self.rho = rho
        self.lr = lr
        self.v = 0
        self.u = 0
        self.eps = eps

    def step(self, var, vargrad):
        
        v = self.v
        u = self.u
        lr = self.lr
        rho = self.rho
        eps = self.eps
        
        v = rho*v + (1-rho)*vargrad**2
        
        self.v = v

        dw = np.sqrt(u+eps)/np.sqrt(v+eps) *vargrad

        u = rho*u + (1-rho)*dw**2
        var = var - lr*dw

        return var
