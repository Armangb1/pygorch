from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, lr=0.001, betas = (0.9, 0.999), eps = 1e-8):
        self.m = 0
        self.v = 0
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.k = 0

        
    def step(self, var, vargrad):
        self.k = self.k+1
        m = self.m
        v = self.v 
        lr = self.lr 
        betas = self.betas 
        eps = self.eps 

        beta1 = betas[0]
        beta2 = betas[1]
        
        m = beta1*m + (1-beta1)*vargrad
        self.m = m

        v = beta2*v + (1-beta2)*vargrad**2
        self.v = v

        m = m/(1-beta1**self.k)
        v = v/(1-beta2**self.k)

        var = var - lr/(np.sqrt(v)+eps) * m

        return var



        
        


        






