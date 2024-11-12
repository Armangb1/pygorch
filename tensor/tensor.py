import numpy as np

class Tensor:
    
    def __init__(self, value, requires_grad:bool = False) -> None:
        if type(value) is not np.ndarray:
            value = np.array(value)
        self.value = value
        self.shape = value.shape
        self.grad = None
        self.requires_grad = requires_grad
        self._grad_fn = None

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.value.shape ==[1] or self.value.shape ==() or self.value.shape ==(1,1) :
                value = np.ones_like(self.value)
                gradient = Tensor(value,requires_grad=True)
            else:
                raise RuntimeError("Gradient argument must be specified for non-scalar tensors.")
                
            
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad += gradient
        

        if self._grad_fn is not None:
            grads = self._grad_fn.backward(gradient)
            for tensor, grad in zip(self._grad_fn.input, grads):
                if isinstance(tensor, Tensor):
                    tensor.backward(grad)

    def zero_grad(self):
        if not self.requires_grad:
            return
        
        self.grad = None
        if self._grad_fn is not None:
            for tensor in self._grad_fn.input:
                if isinstance(tensor, Tensor):
                    tensor.zero_grad()

    def is_grad_eneable(self):
        if self.requires_grad:
            return True
        else:
            return False
    
    def set_grad_disable(self):
        self.requires_grad = False

    def reshape(self, *argv):
        value = self.value.copy()
        try:
            value = value.reshape(argv)
        except ValueError as e:
            raise ValueError(f"Cannot reshape array of size {value.size} into shape {argv}") from e
        return Tensor(value, self.requires_grad)
    
    def copy(self):
        value = self.value.copy()
        requires_grad = self.requires_grad
        return Tensor(value, requires_grad)
        
    def transpose(self):
        value = self.value.copy()
        c = Tensor(value.T,self.requires_grad)
        if c.requires_grad:
            c._grad_fn = TransposeBackward(self)
        return c
    
    def __getitem__(self,idx):
        value = self.value[idx]
        c = Tensor(value,requires_grad=self.requires_grad)
        if c.requires_grad:
            c._grad_fn = GetItemBackward(self,idx)
        return c
    
    def __setitem__(self, key, value):
        self.value[key] =  value

    def __neg__(self):
        c = Tensor(-self.value, requires_grad=self.requires_grad)
        if c.requires_grad:
            c._grad_fn=NegBackward(self)
        return c
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        c =  Tensor(self.value+other.value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = AddBackward(self, other)
        return c

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        c =  Tensor(self.value-other.value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = SubBackward(self, other)
        return c
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        c =  Tensor(self.value*other.value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = MulBackward(self,other)
        return c
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        c =  Tensor(self.value@other.value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = MatMulBackward(self,other)
        return c
    
    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        requires_grad = self.requires_grad
        c =  Tensor(self.value**other.value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = PowBackward(self, other)
        return c
    
    def dot(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        value = self.value.dot(other.value)
        c = Tensor(value, requires_grad)
        if requires_grad:
            c._grad_fn = DotBackward(self, other)
        return c
    
    def sin(self):
        requires_grad = self.requires_grad
        value = np.sin(self.value)
        c =  Tensor(value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = SinBackward(self)
        return c

    def cos(self):
        requires_grad = self.requires_grad
        value = np.cos(self.value)
        c =  Tensor(value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = CosBackward(self)
        return c

    def relu(self):
        requires_grad = self.requires_grad
        value = self.value.copy()
        value=value*(value>0)
        c =  Tensor(value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = ReLuBackward(self)
        return c

    def tanh(self):
        requires_grad = self.requires_grad
        value = np.tanh(self.value)
        c =  Tensor(value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = TanhBackward(self)
        return c


    def sigmoid(self):
        requires_grad = self.requires_grad
        value = 1/(1+np.exp(-self.value))
        c =  Tensor(value,requires_grad=requires_grad)
        if requires_grad:
            c._grad_fn = SigmoidBackward(self)
        return c
    
    def step(self):
        requires_grad = self.requires_grad
        value = self.value.copy()
        value = np.heaviside(value,0)
        if requires_grad:
            c =  Tensor(value,requires_grad=requires_grad)
        return c

    def sum(self , axis=None, keepdims=False):
        """
        Takes a tensor object and returns the sum of its elements along the specified axis.
        
        Args:
        tensor (Tensor): The input tensor whose elements are to be summed.
        axis (int or tuple of ints, optional): Axis or axes along which a sum is performed. 
                                                The default, axis=None, will sum all of the elements of the input tensor.
        keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
        
        Returns:
        Tensor: A tensor containing the sum of the elements along the specified axis.
        """
        if not isinstance(self, Tensor):
            raise ValueError("Input must be a Tensor")
        
        value = np.sum(self.value, axis=axis, keepdims=keepdims)
        return Tensor(value, requires_grad=self.requires_grad)

    
    def __repr__(self) -> str:
        return f"Tensor({self.value})"
    

    @classmethod
    def diag(cls, input):
        value = np.diag(input.value)
        return cls(value)



class AddBackward:
    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.input = [x, y]

    def backward(self, gradient: Tensor) -> list:
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
    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.input = [x, y]

    def backward(self, gradient: Tensor) -> list:
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
    def __init__(self, input: Tensor, idx) -> None:
        self.input = [input]
        self.idx = idx

    def backward(self, gradient: Tensor) -> list:
        grad = np.zeros_like(self.input[0].value)
        np.add.at(grad, self.idx, gradient.value)
        return [Tensor(grad)]
    

class MulBackward:
    def __init__(self, x: Tensor, y: Tensor) -> None:
        self.input = [x, y]

    def backward(self, gradient: Tensor) -> list:
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
    def __init__(self, x:Tensor, y:Tensor) -> None:
        self.input = [x, y]

    def backward(self, gradient: Tensor):
        x = self.input[0]
        y = self.input[1]
        # it is possible that because of its external product the tru equation is:
        # y.transpose().dot(gradient)
        grad = [y.dot(gradient),gradient.dot(x)]
        return grad

class PowBackward:
    def __init__(self, x:Tensor, y) -> None:
        self.input = [x, y]

    def backward(self, gradient: Tensor):
        x = self.input[0]
        y = self.input[1]
        grad_x = y*x**(y-1)
        grad = [gradient.dot(grad_x)]
        return grad

class NegBackward:
    def __init__(self, input:Tensor) -> None:
        self.input = [input]

    def backward(self, gradient:Tensor) -> Tensor:
        grad = [-gradient]
        return grad

class SinBackward:
    def __init__(self, input:Tensor) -> None:
        self.input = [input]

    def backward(self, gradient:Tensor) -> Tensor:
        x = self.input[0]
        grad_x = x.cos()
        grad_x = Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class CosBackward:
    def __init__(self, input:Tensor) -> None:
        self.input = [input]

    def backward(self, gradient:Tensor) -> Tensor:
        x = self.input[0]
        grad_x = -x.cos()
        grad_x = Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class ReLuBackward:
    def __init__(self, input:Tensor) -> None:
        self.input = [input]

    def backward(self, gradient:Tensor) -> Tensor:
        x = self.input[0]
        grad_x = x.step()
        grad_x = Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class TanhBackward:
    def __init__(self, input:Tensor) -> None:
        self.input = [input]

    def backward(self, gradient:Tensor) -> Tensor:
        x = self.input[0]
        grad_x = -x.tanh()**2+1
        grad_x = Tensor(np.diag(grad_x.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad 

class SigmoidBackward:
    def __init__(self, input:Tensor) -> None:
        self.input = [input]

    def backward(self, gradient:Tensor):
        X = self.input[0]
        grad = X.sigmoid()*(-X.sigmoid()+1)
        grad_x = Tensor(np.diag(grad.value.reshape(-1)))
        grad = [gradient.dot(grad_x)]
        return grad
    
class DotBackward:
    def __init__(self, x:Tensor, y:Tensor) -> None:
        self.input = [x, y]

    def backward(self, gradient:Tensor):
        x = self.input[0]
        y = self.input[1]
        # grad = [gradient.dot(y.transpose()),gradient.dot(x)]
        grad = [y.dot(gradient),gradient.dot(x)]
        return grad

class TransposeBackward:
    def __init__(self, input: Tensor) -> None:
        self.input = [input]

    def backward(self, gradient: Tensor) -> list:
        grad = [gradient.transpose()]
        return grad



