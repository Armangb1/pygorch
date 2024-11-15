from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr = 0.001):
        super().__init__()

        self.lr = lr

    def step(self, var, vargrad):
        
        var = var + -vargrad*self.lr
        return var
