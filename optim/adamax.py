from .optimizer import Optimizer
import numpy as np

class Adamax(Optimizer):
    def __init__(self, lr=0.001, betas = (0.9, 0.999), eps = 1e-8):
        super().__init__()
        self.m = 0
        self.u = 0
        self.lr = lr
        self.betas = betas
        self.eps = eps


    def step(self, var, vargrad):
        
        m = self.m
        u = self.u 
        lr = self.lr 
        betas = self.betas 
        eps = self.eps 

        beta1 = betas[0]
        beta2 = betas[1]
        m = beta1*m + (1-beta1)*vargrad
        self.m = m

        u = np.maximum(beta2*u,np.abs(vargrad)+eps)
        self.u = u
        m = m/(1-beta1)

        var = var - lr*m/u

        return var