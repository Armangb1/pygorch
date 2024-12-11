from .optimizer import Optimizer
import numpy as np
class RLS(Optimizer):
    def __init__(self, parameter, lambda_ = 1):
        super().__init__(parameter)
        if len(self.parameters)>1:
            raise ValueError("RLS only supports Linear models and therefore accept only one parameter")
        self.lambda_ = lambda_
        self.P = np.zeros((self.parameters[0].shape[0],self.parameters[0].shape[0]))

            

    def step(self, y, x):
        
        y_d = y.value.T
        x_d = x.value.T
        P = self.P
        lambda_ = self.lambda_
        param = self.parameters[0].value
        
        e = y_d.T - x_d.T@param
        gamma = 1/(lambda_ + x_d.T@P@x_d) * P@x_d
        P = 1/lambda_*(P - gamma@x_d.T@P)
        param += gamma@e

        self.parameters[0].value = param
