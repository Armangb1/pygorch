from .optimizer import Optimizer
import numpy as np 

class RMSprop(Optimizer):
    
    def __init__(self, lr = 0.001, alpha = 0.99, eps = 1e-8):
        super().__init__()

        self.lr = lr
        self.alpha = alpha
        self.v = 0
        self.eps = eps

    def step(self, var, vargrad):
        
        lr = self.lr
        v = self.v
        alpha = self.alpha
        eps = self.eps

        v = alpha*v + (1-alpha)*vargrad**2

        self.v = v

        var = var - lr/(np.sqrt(v)+eps)*vargrad

        return var
