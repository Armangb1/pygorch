from .tensor import Tensor
import numpy as np
from .backward import *


def diag(tensor:Tensor) -> Tensor:
    """
    Takes a tensor object and returns a diagonalized tensor.
    
    Args:
    tensor (torch.Tensor): The input tensor to be diagonalized.
    
    Returns:
    torch.Tensor: The diagonalized tensor.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.diag(tensor.value)
    return Tensor(value, requires_grad=tensor.requires_grad)

def eye(n: int, m: int = None, requires_grad: bool = False) -> Tensor:
    """
    Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.
    
    Args:
    n (int): Number of rows in the output tensor.
    m (int, optional): Number of columns in the output tensor. If None, defaults to n.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A 2-D tensor with ones on the diagonal and zeros elsewhere.
    """
    if m is None:
        m = n
    
    value = np.eye(n, m)
    return Tensor(value, requires_grad=requires_grad)

def ones(*shape, requires_grad=False) -> Tensor:
    """
    Creates a tensor filled with ones.
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with ones.
    """
    value = np.ones(shape)
    return Tensor(value, requires_grad=requires_grad)

def zeros(*shape, requires_grad=False) -> Tensor:
    """
    Creates a tensor filled with zeros.
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with zeros.
    """
    value = np.zeros(shape)
    return Tensor(value, requires_grad=requires_grad)

def rand(*shape, requires_grad=False) -> Tensor:
    """
    Creates a tensor filled with random values from a uniform distribution over [0, 1).
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a uniform distribution over [0, 1).
    """
    value = np.random.rand(*shape)
    return Tensor(value, requires_grad=requires_grad)

def randn(*shape, requires_grad=False) -> Tensor:
    """
    Creates a tensor filled with random values from a standard normal distribution.
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a standard normal distribution.
    """
    value = np.random.randn(*shape)
    return Tensor(value, requires_grad=requires_grad)

def rand_like(tensor: Tensor, requires_grad=False) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the same shape filled with random values from a uniform distribution over [0, 1).
    
    Args:
    tensor (Tensor): The input tensor to create a random tensor from.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a uniform distribution over [0, 1) with the same shape as the input tensor.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.random.rand(*tensor.shape)
    return Tensor(value, requires_grad=requires_grad)

def randn_like(tensor: Tensor, requires_grad=False) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the same shape filled with random values from a standard normal distribution.
    
    Args:
    tensor (Tensor): The input tensor to create a random tensor from.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a standard normal distribution with the same shape as the input tensor.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.random.randn(*tensor.shape)
    return Tensor(value, requires_grad=requires_grad)

def mean(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Takes a tensor object and returns the mean of its elements along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor whose elements are to be averaged.
    axis (int or tuple of ints, optional): Axis or axes along which a mean is performed. 
                                            The default, axis=None, will compute the mean of all the elements of the input tensor.
    keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
    
    Returns:
    Tensor: A tensor containing the mean of the elements along the specified axis.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.mean(tensor.value, axis=axis, keepdims=keepdims)
    return Tensor(value, requires_grad=tensor.requires_grad)

def var(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Takes a tensor object and returns the variance of its elements along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor whose elements' variance is to be computed.
    axis (int or tuple of ints, optional): Axis or axes along which a variance is performed. 
                                            The default, axis=None, will compute the variance of all the elements of the input tensor.
    keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
    
    Returns:
    Tensor: A tensor containing the variance of the elements along the specified axis.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.var(tensor.value, axis=axis, keepdims=keepdims)
    return Tensor(value, requires_grad=tensor.requires_grad)

def median(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Takes a tensor object and returns the median of its elements along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor whose elements' median is to be computed.
    axis (int or tuple of ints, optional): Axis or axes along which a median is performed. 
                                            The default, axis=None, will compute the median of all the elements of the input tensor.
    keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
    
    Returns:
    Tensor: A tensor containing the median of the elements along the specified axis.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.median(tensor.value, axis=axis, keepdims=keepdims)
    return Tensor(value, requires_grad=tensor.requires_grad)

def sum(tensor: Tensor, axis=None, keepdims=False) -> Tensor:
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
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sum(tensor.value, axis=axis, keepdims=keepdims)
    return Tensor(value, requires_grad=tensor.requires_grad)


def ones_like(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor filled with ones having the same shape.
    
    Args:
    tensor (Tensor): The input tensor to create a ones tensor from.
    
    Returns:
    Tensor: A tensor filled with ones having the same shape as the input tensor.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.ones_like(tensor.value)
    return Tensor(value, requires_grad=tensor.requires_grad)

def exp(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the exponential function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the exponential function to.
    
    Returns:
    Tensor: A tensor with the exponential function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.exp(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = ExpBackward(tensor)
    return out



def tanh(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the hyperbolic tangent applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply tanh to.
    
    Returns:
    Tensor: A tensor with the tanh function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.tanh(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = TanhBackward(tensor)
    return out

def sinh(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the hyperbolic sine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply sinh to.
    
    Returns:
    Tensor: A tensor with the sinh function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sinh(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SinhBackward
    return out

def cosh(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the hyperbolic cosine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply cosh to.
    
    Returns:
    Tensor: A tensor with the cosh function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.cosh(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = CoshBackward(tensor)
    return out



def cos(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the cosine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply cosine to.
    
    Returns:
    Tensor: A tensor with the cosine function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.cos(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = CosBackward(tensor)
    return out

def sin(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the sine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply sine to.
    
    Returns:
    Tensor: A tensor with the sine function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sin(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = ReLuBackward(tensor)
    return out


def tan(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the tangent applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply tangent to.
    
    Returns:
    Tensor: A tensor with the tangent function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.tan(tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = TanBackward(tensor)
    return out

def sigmoid(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the sigmoid function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply sigmoid to.
    
    Returns:
    Tensor: A tensor with the sigmoid function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = 1 / (1 + np.exp(-tensor.value))
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SigmoidBackward(tensor)
    return out

def relu(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the ReLU function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply ReLU to.
    
    Returns:
    Tensor: A tensor with the ReLU function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.maximum(0, tensor.value)
    out = Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = ReLuBackward(tensor)
    return out

def step(tensor: Tensor) -> Tensor:
    """
    Takes a tensor object and returns a tensor with the step function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the step function to.
    
    Returns:
    Tensor: A tensor with the step function applied element-wise.
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.where(tensor.value > 0, 1, 0)
    return Tensor(value, requires_grad=tensor.requires_grad)

def maximum(input: Tensor, other: Tensor) -> Tensor:
    """
    Takes two tensor objects and returns a tensor with the element-wise maximum.
    
    Args:
    input (Tensor): The first input tensor.
    other (Tensor): The second input tensor.
    
    Returns:
    Tensor: A tensor with the element-wise maximum.
    """
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise ValueError("Inputs must be Tensors")
    
    value = np.maximum(input.value, other.value)
    out = Tensor(value, requires_grad=input.requires_grad or other.requires_grad)
    if out.requires_grad:
        out._grad_fn = MaximumBackward(input, other)
    return out

def minimum(input: Tensor, other: Tensor) -> Tensor:
    """
    Takes two tensor objects and returns a tensor with the element-wise minimum.
    
    Args:
    input (Tensor): The first input tensor.
    other (Tensor): The second input tensor.
    
    Returns:
    Tensor: A tensor with the element-wise minimum.
    """
    if not isinstance(input, Tensor) or not isinstance(other, Tensor):
        raise ValueError("Inputs must be Tensors")
    
    value = np.minimum(input.value, other.value)
    out = Tensor(value, requires_grad=input.requires_grad or other.requires_grad)
    if out.requires_grad:
        out._grad_fn = MinimumBackward(input, other)
    return out