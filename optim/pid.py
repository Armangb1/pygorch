from .optimizer import Optimizer
import numpy as np

class PID(Optimizer):
    def __init__(self, kp = 0.001, ki = 0.0001, kd=0.0001, betas=(0.9,0.9)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i = 0
        self.d = 0
        self.betas = betas
        self.preGrad = 0

        
    def step(self, var, vargrad):
        kp = self.kp
        ki = self.ki
        kd = self.kd
        i = self.i
        d = self.d
        preGrad = self.preGrad
        betas = self.betas

        beta_1 = betas[0]
        beta_2 = betas[1]

        i = beta_1*i + vargrad
        d = beta_2*d +vargrad-preGrad
        self.preGrad =vargrad

        self.i = i
        self.d = d

        var = var - kp*vargrad - ki*i- kd*d        
        


        return var



        
        


        






