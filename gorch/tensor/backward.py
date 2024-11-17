from typing import TYPE_CHECKING
import gorch
if TYPE_CHECKING:
    from .tensor import Tensor
import numpy as np

__all__ = [
    "AddBackward",
    "SubBackward",
    "GetItemBackward",
    "MulBackward",
    "MatMulBackward",
    "DotBackward",
    "PowBackward",
    "SumBackward",
    "NegBackward",
    "TransposeBackward",
    "SinBackward",
    "CosBackward",
    "TanBackward",
    "ExpBackward",
    "SinhBackward",
    "CoshBackward",
    "TanhBackward",
    "SigmoidBackward",
    "ReLuBackward",
    "MaximumBackward",
    "MinimumBackward",
]

class AddBackward:
    def __init__(self, x: 'Tensor', y: 'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor') -> list:
        x, y = self.input
        grad_x = gradient.transpose()
        grad_y = gradient.transpose()

        # Handle broadcasting
        while len(grad_x.shape) > len(x.shape):
            grad_x = grad_x.sum(axis=0)
        while len(grad_y.shape) > len(y.shape):
            grad_y = grad_y.sum(axis=0)

        for axis in range(-1, -len(x.shape) - 1, -1):
            if x.shape[axis] == 1:
                grad_x = grad_x.sum(axis=axis, keepdims=True)
        for axis in range(-1, -len(y.shape) - 1, -1):
            if y.shape[axis] == 1:
                grad_y = grad_y.sum(axis=axis, keepdims=True)

        return [grad_x.transpose(), grad_y.transpose()]
    
class SubBackward:
    def __init__(self, x: 'Tensor', y: 'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor') -> list:
        x, y = self.input
        grad_x = gradient.transpose()
        grad_y = gradient.transpose()

        # Handle broadcasting
        while len(grad_x.shape) > len(x.shape):
            grad_x = grad_x.sum(axis=0)
        while len(grad_y.shape) > len(y.shape):
            grad_y = grad_y.sum(axis=0)

        for axis, (dim_x, dim_y) in enumerate(zip(x.shape, y.shape)):
            if dim_x == 1:
                grad_x = grad_x.sum(axis=axis, keepdims=True)
            if dim_y == 1:
                grad_y = grad_y.sum(axis=axis, keepdims=True)

        return [grad_x.transpose(), -grad_y.transpose()]

class GetItemBackward:
    def __init__(self, input: 'Tensor', idx) -> None:
        self.input = [input]
        self.idx = idx

    def backward(self, gradient: 'Tensor') -> list:
        grad = np.zeros_like(self.input[0].value)
        np.add.at(grad, self.idx, gradient.value)
        return [gorch.Tensor(grad)]
    
class MulBackward:
    def __init__(self, x: 'Tensor', y: 'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor') -> list:
        x, y = self.input
        gradient = gradient.transpose()
        grad_x = gradient*y
        grad_y = gradient*x
        # Handle broadcasting
        while len(grad_x.shape) > len(x.shape):
            grad_x = grad_x.sum(axis=0)
        while len(grad_y.shape) > len(y.shape):
            grad_y = grad_y.sum(axis=0)

        for axis in range(-1, -len(x.shape) - 1, -1):
            if x.shape[axis] == 1:
                grad_x = grad_x.sum(axis=axis, keepdims=True)
        for axis in range(-1, -len(y.shape) - 1, -1):
            if y.shape[axis] == 1:
                grad_y = grad_y.sum(axis=axis, keepdims=True)
        grad_x = grad_x.transpose()
        grad_y = grad_y.transpose()
        return [grad_x, grad_y]
    
class MatMulBackward:
    def __init__(self, x:'Tensor', y:'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor'):
        x = self.input[0]
        y = self.input[1]
        # it is possible that because of its external product the tru equation is:
        # y.transpose().dot(gradient)
        grad = [y.dot(gradient),gradient.dot(x)]
        return grad

class DotBackward:
    def __init__(self, x:'Tensor', y:'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient:'Tensor'):
        x = self.input[0]
        y = self.input[1]
        # grad = [gradient.dot(y.transpose()),gradient.dot(x)]
        grad = [y.dot(gradient),gradient.dot(x)]
        return grad
    
class PowBackward:
    def __init__(self, x:'Tensor', y) -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor'):
        x = self.input[0]
        y = self.input[1]
        grad_x = y*x**(y-1)
        grad = [gradient.dot(grad_x)]
        return grad

class SumBackward:
    def __init__(self, input: 'Tensor', axis=None, keepdims=False) -> None:
        self.input = [input]
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = gradient

        if self.axis is None:
            grad_x = np.broadcast_to(grad_x, x.shape)
        else:
            if not self.keepdims:
                grad_x = np.expand_dims(grad_x, axis=self.axis)
            grad_x = np.broadcast_to(grad_x, x.shape)

        return [gorch.Tensor(grad_x)]

class NegBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        grad = [-gradient]
        return grad
  
class TransposeBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> list:
        grad = [gradient.transpose()]
        return grad

class SinBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.cos()
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class CosBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = -x.cos()
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class TanBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = 1-x.tan()**2
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad

class ExpBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.exp()
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad

class SinhBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.cosh()
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad

class CoshBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.sinh()
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class TanhBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = -x.tanh()**2+1
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad 

class SigmoidBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor'):
        X = self.input[0]
        grad = X.sigmoid()*(-X.sigmoid()+1)
        grad_x = gorch.Tensor(np.diag(grad.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad

class ReLuBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.step()
        grad_x = gorch.Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class MaximumBackward:
    def __init__(self, x:'Tensor', y:'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient:'Tensor'):
        x = self.input[0]
        y = self.input[1]
        grad_x = x.value>y.value
        grad_y = ~grad_x
        grad_x = grad_x.astype(float)
        grad_y = grad_y.astype(float)
        grad_x = gorch.Tensor(np.diag(grad_x))
        grad_y = gorch.Tensor(np.diag(grad_y))
        
        grad = [gradient.dot(grad_x),gradient.dot(grad_y)]
        return grad
    
class MinimumBackward:
    def __init__(self, x:'Tensor', y:'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient:'Tensor'):
        x = self.input[0]
        y = self.input[1]
        grad_x = x.value<y.value
        grad_y = ~grad_x
        grad_x = grad_x.astype(float)
        grad_y = grad_y.astype(float)
        grad_x = gorch.Tensor(np.diag(grad_x))
        grad_y = gorch.Tensor(np.diag(grad_y))
        
        grad = [gradient.dot(grad_x),gradient.dot(grad_y)]
        return grad


