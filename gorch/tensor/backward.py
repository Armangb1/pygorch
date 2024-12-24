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
    "DivBackward",
    "MatMulBackward",
    "DotBackward",
    "PowBackward",
    "SumBackward",
    "NegBackward",
    "TransposeBackward",
    "ReshapeBackward",
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
    "MeanBackward",
    "SqrtBackward",
    "LogBackward",
    "AbsBackward",
    "SumBackward",
    "MaxBackward",
    "MinBackward",
    "NormBackward",
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

class DivBackward:
    def __init__(self, x: 'Tensor', y: 'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor') -> list:
        x, y = self.input
        gradient = gradient.transpose()
        grad_x = gradient / y
        grad_y = -gradient * x / (y * y)
        
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
        # grad_x = gorch.diag(grad_x.reshape(-1))
        grad = [gradient*grad_x.transpose()]
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
        grad = [gradient*grad_x.transpose()]
        return grad
    
class CosBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = -x.sin()
        grad = [gradient*grad_x.transpose()]
        return grad
    
class TanBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = 1-x.tan()**2
        grad = [gradient*grad_x.transpose()]
        return grad

class ExpBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.exp()
        grad = [gradient*grad_x.transpose()]
        return grad

class SinhBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.cosh()
        grad = [gradient*grad_x.transpose()]
        return grad

class CoshBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.sinh()
        grad = [gradient*grad_x.transpose()]
        return grad
    
class TanhBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = -x.tanh()**2+1
        grad = [gradient*grad_x.transpose()]
        return grad 

class SigmoidBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor'):
        X = self.input[0]
        grad = X.sigmoid()*(-X.sigmoid()+1)
        grad = [gradient*grad.transpose()]
        return grad

class ReLuBackward:
    def __init__(self, input:'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient:'Tensor') -> 'Tensor':
        x = self.input[0]
        grad_x = x.step()
        grad = [gradient*grad_x.transpose()]
        return grad
    
class MaximumBackward:
    def __init__(self, x: 'Tensor', y: 'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor') -> list:
        x, y = self.input
        grad_x = gradient.transpose() * (x.value >= y.value)
        grad_y = gradient.transpose() * (x.value < y.value)

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
    
class MinimumBackward:
    def __init__(self, x: 'Tensor', y: 'Tensor') -> None:
        self.input = [x, y]

    def backward(self, gradient: 'Tensor') -> list:
        x, y = self.input
        grad_x = gradient.transpose() * (x.value <= y.value)
        grad_y = gradient.transpose() * (x.value > y.value)

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

class MeanBackward:
    def __init__(self, input: 'Tensor', axis=None, keepdims=False) -> None:
        self.input = [input]
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = gradient.transpose()

        if self.axis is None:
            grad_x.value = np.broadcast_to(grad_x.value, x.shape) / x.value.size
        else:
            if not self.keepdims:
                grad_x.value = np.expand_dims(grad_x.value, axis=self.axis)
            grad_x.value = np.broadcast_to(grad_x.value, x.shape) / x.value.shape[self.axis]

        return [grad_x.transpose()]
    
class SqrtBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = 0.5 / x.sqrt()
        grad = [gradient * grad_x]
        return grad
        
class LogBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = gorch.Tensor(1) / x
        grad = [gradient * grad_x.transpose()]
        return grad

class AbsBackward:
    def __init__(self, input: 'Tensor') -> None:
        self.input = [input]

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = x.value / np.abs(x.value)
        grad = [gradient * gorch.Tensor(grad_x.T)]
        return grad
    
class SumBackward:
    def __init__(self, input: 'Tensor', axis=None, keepdims=False) -> None:
        self.input = [input]
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = gradient.transpose().value

        if self.axis is None:
            grad_x = np.broadcast_to(grad_x, x.shape)
        else:
            if not self.keepdims:
                grad_x = np.expand_dims(grad_x, axis=self.axis)
            grad_x = np.broadcast_to(grad_x, x.shape)

        return [gorch.Tensor(grad_x.T)]
    
class MaxBackward:
    def __init__(self, input: 'Tensor', axis=None, keepdims=False) -> None:
        self.input = [input]
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = np.zeros_like(x.value)
        max_indices = np.argmax(x.value, axis=self.axis, keepdims=True)
        np.put_along_axis(grad_x, max_indices, gradient.value, axis=self.axis)

        if not self.keepdims and self.axis is not None:
            grad_x = np.expand_dims(grad_x, axis=self.axis)

        return [gorch.Tensor(grad_x)]
    
class MinBackward:
    def __init__(self, input: 'Tensor', axis=None, keepdims=False) -> None:
        self.input = [input]
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        grad_x = np.zeros_like(x.value)
        min_indices = np.argmin(x.value, axis=self.axis, keepdims=True)
        np.put_along_axis(grad_x, min_indices, gradient.value, axis=self.axis)

        if not self.keepdims and self.axis is not None:
            grad_x = np.expand_dims(grad_x, axis=self.axis)

        return [gorch.Tensor(grad_x)]

class NormBackward:
    def __init__(self, input: 'Tensor', ord=2, axis=None, keepdims=False) -> None:
        self.input = [input]
        self.ord = ord
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, gradient: 'Tensor') -> list:
        x = self.input[0]
        norm = np.linalg.norm(x.value, ord=self.ord, axis=self.axis, keepdims=True)
        grad_x = np.zeros_like(x.value)

        if self.ord == 2:
            grad_x = x.value / norm
        elif self.ord == 1:
            grad_x = np.sign(x.value)
        else:
            grad_x = np.power(np.abs(x.value), self.ord - 1) * np.sign(x.value) / norm**(self.ord - 1)

        if self.axis is not None and not self.keepdims:
            grad_x = np.expand_dims(grad_x, axis=self.axis)

        grad_x = grad_x.T * gradient.value
        return [gorch.Tensor(grad_x)]

class ReshapeBackward:
    def __init__(self, input: 'Tensor', shape) -> None:
        self.input = [input]
        self.shape = shape

    def backward(self, gradient: 'Tensor') -> list:
        grad_x = gradient.transpose().value
        grad_x = grad_x.reshape(self.input[0].shape)
        if grad_x.ndim == 1:
            grad_x = grad_x.reshape(1,-1)
        return [gorch.Tensor(grad_x.T)]