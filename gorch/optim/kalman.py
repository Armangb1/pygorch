from .optimizer import Optimizer
import numpy as np 
from gorch.nn.functional import jacobian, devectorize_parameters, vectorize_parameters, zero_grad
class KalmanFilter:
    def __init__(self, A, B, C, D, Q, R, x = None, P = None) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        if x is None:
            self.x = np.zeros((A.shape[0], 1))
        else:
            self.x = x
        if P is None:
            self.P = np.random.rand(*A.shape)
        else:
            self.P = P
        self.Q = Q
        self.R = R


    def step(self, y, u):
        x = self.x
        P = self.P
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        Q = self.Q
        R = self.R

        x = A@x + B@u
        P = A@P@A.T + Q
        K = P@C.T@(np.linalg.inv(C@P@C.T + R))
        x = x + K@(y - C@x - D@u)
        P = (np.eye(A.shape[0]) - K@C)@P
        self.x = x
        self.P = P
        return x
    

class EKFOptimizer(Optimizer):
    def __init__(self, parameters, Q = None, R = None):
        super().__init__(parameters)
        x, _ = vectorize_parameters(self.parameters)
        self.P = np.random.randn(x.shape[0],x.shape[0])
        self.x = x.reshape(-1,1)
        dx = np.zeros_like(x)
        self.dx = dx.reshape(-1,1)
        if Q is not None:
            if isinstance(Q, float):
                self.Q = np.abs(np.random.randn(x.shape[0],x.shape[0]))*Q
            else:
                self.Q = Q
        else:
            # self.Q = np.eye(x.shape[0],x.shape[0])*0.01
            self.Q = np.abs(np.random.randn(x.shape[0],x.shape[0]))*0.001
        if R is not None:
            if isinstance(R, float):
                dim = self.parameters[-1].shape[-1]
                self.R = np.abs(np.random.randn(dim, dim))*R
            else:
                self.R = R
        else:
            dim = self.parameters[-1].shape[-1]
            # self.R = np.eye(dim, dim)*0.1
            self.R = np.abs(np.random.randn(dim, dim))*0.01
        
    
    def step(self,inputs, y, model):
        x,_ = vectorize_parameters(model)
        x = x.reshape(-1,1)
        A = np.eye(x.shape[0], x.shape[0])
        B = np.array([[0]])
        D = np.array([[0]])
        C = jacobian(model, inputs)
        u = np.array([[0]])
        Q = self.Q
        R = self.R
        P = self.P
        dx = self.dx
        y_pred = model(inputs)
        # dy = y.value.T - y_pred.value.T
        
        e = y.value.T - y_pred.value.T
        P = P+Q
        S = R + C@P@C.T
        K = P@C.T@np.linalg.inv(S)
        P = P-K@C@P
        x = x + K@e
        self.P = P
        # self.P = P
        # kalman = KalmanFilter(A,B,C,D,Q,R,x,P)
        # x = kalman.step(dy, u)
        # self.P = kalman.P
        # self.dx = dx
        devectorize_parameters(model, x)