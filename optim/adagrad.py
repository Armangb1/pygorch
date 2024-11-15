from .optimizer import Optimizer
import numpy as np
# this is a basic implementation of
# Adagrad. this optimizer can be improved with
# weigth decay rate an lr decay
class Adagrad(Optimizer):
    def __init__(self,lr=0.03):
        self.lr = lr
        self.S = 0
        self.eps = 1e-7

    def step(self, var, vargrad):
        
        self.S = self.S + vargrad**2
        var = var - self.lr/(np.sqrt(self.S)+ self.eps) *vargrad
        return var
        

    