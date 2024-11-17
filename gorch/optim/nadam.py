from .optimizer import Optimizer
import numpy as np


# TODO: check the step
class NAdam(Optimizer):
    def __init__(self, lr=0.001, betas = (0.9, 0.999), eps = 1e-8, psi = 0.004):
        self.m = 0
        self.v = 0
        self.p = 1
        self.mu = 0
        self.t = 1
        self.psi = psi
        self.lr = lr
        self.betas = betas
        self.eps = eps

        
    def step(self, var, vargrad):
        m = self.m
        v = self.v
        p = self.p
        lr = self.lr 
        betas = self.betas 
        eps = self.eps 
        beta1 = betas[0]
        beta2 = betas[1]
        
        m = beta1*m + (1-beta1)*vargrad
        self.m = m

        v = beta2*v + (1-beta2)*vargrad**2
        self.v = v

        mu_t = 1-0.5*(0.96)**(self.psi*self.t)
        mu_t1 = 1-0.5*(0.96)**(self.psi*(self.t+1))

        p = 1- mu_t1*(1-p)
        m = mu_t1*m/p + (1-mu_t)*vargrad/self.p

        self.p = p
        v = v/(1-beta2)

        var = var - lr/(np.sqrt(v)+eps) * m

        return var



        
        


        






