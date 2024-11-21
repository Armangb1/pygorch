import numpy as np
from .backward import *
import gorch.tensor.methods as methods


class Tensor:
    
    def __init__(self, value, requires_grad:bool = False) -> None:
        if not isinstance(value, np.ndarray):
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
            self.grad = gradient.transpose()
        else:
            self.grad += gradient.transpose()
        

        if self._grad_fn is not None:
            grads = self._grad_fn.backward(gradient)
            for tensor, grad in zip(self._grad_fn.input, grads):
                if isinstance(tensor, Tensor):
                    tensor.backward(grad)

    def detach(self):
        return Tensor(self.value.copy(), requires_grad=False)
    
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
    
    def __radd__(self, other):
        return self.__add__(other)

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

    def copy(self):
        value = self.value.copy()
        requires_grad = self.requires_grad
        return Tensor(value, requires_grad)
        
    def reshape(self, *argv):
        return methods.reshape(self, argv)

    def transpose(self):
        return methods.transpose(self)

    def dot(self, other):
        return methods.dot(self, other)
    
    def sin(self):
        return methods.sin(self)

    def cos(self):
        return methods.cos(self)

    def tan(self):
        return methods.tan(self)
    
    def exp(self):
        return methods.exp(self)
    
    def relu(self):
        return methods.relu(self)

    def sinh(self):
        return methods.sinh(self)
    
    def cosh(self):
        return methods.cosh(self)
    
    def tanh(self):
        return methods.tanh(self)

    def sigmoid(self):
        return methods.sigmoid(self)
    
    def step(self):
        return methods.step(self)
    
    def sum(self , axis=None, keepdims=False):
        return methods.sum(self, axis, keepdims)
    
    def abs(self):
        return methods.abs(self)
    
    def sqrt(self):
        return methods.sqrt(self)
    
    def log(self):
        return methods.log(self)
    
    def __repr__(self) -> str:
        return f"Tensor({self.value})"