from .optimizer import Optimizer


class Nestrov(Optimizer):
    def __init__(self, grad_fn =None, lr = 0.001, beta= 0.9):
        super().__init__()
        self.lr = lr
        self.beta = beta
        self.v = 0


    def step(self, var, grad_fn):
        v = self.v
        beta = self.beta
        lr =self.lr
        vargrad = grad_fn(var-lr*v)
        v = beta*v + (1-beta)*vargrad.T
        self.v = v

        var = var - lr*v

        return var



