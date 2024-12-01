from .optimizer import Optimizer
import numpy as np
from gorch.nn.functional import jacobian, devectorize_parameters

class LM(Optimizer):
    def __init__(self, parameter, lambd = 0.001):
        super().__init__(parameter)
        self.lambda_ = lambd
    

    def step(self, model, x, y):
        y_pred = model(x)
        jac = jacobian(model, x)
        H = jac.T@jac
        g = jac.T@(y_pred-y)
        H = H + self.lambda_*np.eye(H.shape[0])
        dtheta = np.linalg.solve(H, g)
        devectorize_parameters(model, dtheta)
