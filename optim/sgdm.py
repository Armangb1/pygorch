from .optimizer import Optimizer


class SGDM(Optimizer):
    def __init__(self, lr = 0.001, beta = 0.9):
        super().__init__()
        self.lr = lr
        self.m = 0
        self.beta = beta


    def step(self, var, vargrad):
        beta = self.beta
        self.m = beta * self.m + (1-beta)*vargrad
        var = var - self.lr*self.m
        return var


             
        
        