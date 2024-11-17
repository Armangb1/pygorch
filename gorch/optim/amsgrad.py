from .optimizer import Optimizer
import numpy as np

# TODO: make v ok with initializer
class AMSGrad(Optimizer):
    def __init__(self, lr=0.001, betas = (0.9, 0.999), eps = 1e-8):
        self.m = 0
        self.v = None
        self.lr = lr
        self.betas = betas
        self.eps = eps

        
    def step(self, var, vargrad):

        if self.v is None:
            self.v = np.zeros_like(var)

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
        v = np.maximum(self.v,v)

        self.v = v

        var = var - lr/(np.sqrt(v)+eps) * m

        return var



        
        


        






