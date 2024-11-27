from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas = (0.9, 0.999), eps = 1e-8):
        super().__init__(parameters)
        self.m = [np.zeros_like(p.value) for p in self.parameters]
        self.v = [np.zeros_like(p.value) for p in self.parameters]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.k = 0

        
    def step(self):
        self.k = self.k+1

        lr = self.lr 
        betas = self.betas 
        eps = self.eps 

        beta1 = betas[0]
        beta2 = betas[1]
        
        for i, param in enumerate(self.parameters):
            m = self.m[i]
            v = self.v[i]
            m = beta1*m + (1-beta1)*param.grad.value
            self.m[i] = m

            v = beta2*v + (1-beta2)*param.grad.value**2
            self.v[i] = v

            m_hat = m/(1-beta1**self.k)
            v_hat = v/(1-beta2**self.k)

            param.value -= lr/(np.sqrt(v_hat)+eps) * m_hat




        
        


        






