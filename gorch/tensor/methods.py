from typing import TYPE_CHECKING
import gorch
if TYPE_CHECKING:
    from .tensor import Tensor
import numpy as np
from .backward import *

def reshape(tensor: 'Tensor', *new_shape) -> 'Tensor':
    """
    Takes a tensor object and returns a reshaped tensor.
    
    Args:
    tensor (Tensor): The input tensor to be reshaped.
    new_shape (sequence of ints): The new shape for the tensor.
    
    Returns:
    Tensor: The reshaped tensor.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.reshape(tensor.value, new_shape[0])
    return gorch.Tensor(value, requires_grad=tensor.requires_grad)

def transpose(tensor: 'Tensor', axes=None) -> 'Tensor':
    """
    Takes a tensor object and returns a transposed tensor.
    
    Args:
    tensor (Tensor): The input tensor to be transposed.
    axes (tuple of ints, optional): By default, reverse the dimensions, otherwise permute the axes according to the values given.
    
    Returns:
    Tensor: The transposed tensor.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.transpose(tensor.value, axes=axes)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = TransposeBackward(tensor)
    return out

def dot(tensor1: 'Tensor', tensor2: 'Tensor') -> 'Tensor':
    """
    Takes two tensor objects and returns their dot product.
    
    Args:
    tensor1 (Tensor): The first input tensor.
    tensor2 (Tensor): The second input tensor.
    
    Returns:
    Tensor: The dot product of the input tensors.
    """
    if not isinstance(tensor1, gorch.Tensor) or not isinstance(tensor2, gorch.Tensor):
        raise ValueError("Inputs must be Tensors")
    
    value = np.dot(tensor1.value, tensor2.value)
    out = gorch.Tensor(value, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    if out.requires_grad:
        out._grad_fn = DotBackward(tensor1, tensor2)
    return out

def diag(tensor:'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a diagonalized tensor.
    
    Args:
    tensor (torch.Tensor): The input tensor to be diagonalized.
    
    Returns:
    torch.Tensor: The diagonalized tensor.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.diag(tensor.value)
    return gorch.Tensor(value, requires_grad=tensor.requires_grad)

def eye(*shape, requires_grad: bool = False) -> 'Tensor':
    """
    Creates a tensor with ones on the diagonal and zeros elsewhere.
    
    Args:
    shape (int...): Dimensions of the output tensor. If more than 2 dimensions are provided, 
                    the function will create a batch of 2-D identity matrices.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor with ones on the diagonal and zeros elsewhere.
    """
    if len(shape) <= 2:
        value = np.eye(*shape)
    if len(shape) > 2:
        iter = int(np.array(shape[:-2]).sum())
        value = np.concatenate([np.eye(shape[-2], shape[-1]) for i in range(iter)], axis=0)
        value = value.reshape(*shape)

    return gorch.Tensor(value, requires_grad=requires_grad)

def ones(*shape, requires_grad=False) -> 'Tensor':
    """
    Creates a tensor filled with ones.
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with ones.
    """
    value = np.ones(*shape)
    return gorch.Tensor(value, requires_grad=requires_grad)

def zeros(*shape, requires_grad=False) -> 'Tensor':
    """
    Creates a tensor filled with zeros.
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with zeros.
    """
    value = np.zeros(shape)
    return gorch.Tensor(value, requires_grad=requires_grad)

def rand(*shape, requires_grad=False) -> 'Tensor':
    """
    Creates a tensor filled with random values from a uniform distribution over [0, 1).
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a uniform distribution over [0, 1).
    """
    value = np.random.rand(*shape)
    return gorch.Tensor(value, requires_grad=requires_grad)

def randn(*shape, requires_grad=False) -> 'Tensor':
    """
    Creates a tensor filled with random values from a standard normal distribution.
    
    Args:
    shape (int...): Dimensions of the output tensor.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a standard normal distribution.
    """
    value = np.random.randn(*shape)
    return gorch.Tensor(value, requires_grad=requires_grad)

def rand_like(tensor: 'Tensor', requires_grad=False) -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the same shape filled with random values from a uniform distribution over [0, 1).
    
    Args:
    tensor (Tensor): The input tensor to create a random tensor from.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a uniform distribution over [0, 1) with the same shape as the input tensor.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.random.rand(*tensor.shape)
    return gorch.Tensor(value, requires_grad=requires_grad)

def randn_like(tensor: 'Tensor', requires_grad=False) -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the same shape filled with random values from a standard normal distribution.
    
    Args:
    tensor (Tensor): The input tensor to create a random tensor from.
    requires_grad (bool, optional): If autograd should record operations on the returned tensor.
    
    Returns:
    Tensor: A tensor filled with random values from a standard normal distribution with the same shape as the input tensor.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.random.randn(*tensor.shape)
    return gorch.Tensor(value, requires_grad=requires_grad)

def mean(tensor: 'Tensor', axis=None, keepdims=False) -> 'Tensor':
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
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.mean(tensor.value, axis=axis, keepdims=keepdims)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = MeanBackward(tensor,axis,keepdims)
    return out

def var(tensor: 'Tensor', axis=None, keepdims=False) -> 'Tensor':
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
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.var(tensor.value, axis=axis, keepdims=keepdims)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    return out

def median(tensor: 'Tensor', axis=None, keepdims=False) -> 'Tensor':
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
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.median(tensor.value, axis=axis, keepdims=keepdims)
    return gorch.Tensor(value, requires_grad=tensor.requires_grad)

def sum(tensor: 'Tensor', axis=None, keepdims=False) -> 'Tensor':
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
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sum(tensor.value, axis=axis, keepdims=keepdims)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SumBackward(tensor,axis,keepdims)
    return out

def ones_like(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor filled with ones having the same shape.
    
    Args:
    tensor (Tensor): The input tensor to create a ones tensor from.
    
    Returns:
    Tensor: A tensor filled with ones having the same shape as the input tensor.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.ones_like(tensor.value)
    return gorch.Tensor(value, requires_grad=tensor.requires_grad)

def exp(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the exponential function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the exponential function to.
    
    Returns:
    Tensor: A tensor with the exponential function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.exp(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = ExpBackward(tensor)
    return out



def tanh(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the hyperbolic tangent applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply tanh to.
    
    Returns:
    Tensor: A tensor with the tanh function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.tanh(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = TanhBackward(tensor)
    return out

def sinh(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the hyperbolic sine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply sinh to.
    
    Returns:
    Tensor: A tensor with the sinh function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sinh(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SinhBackward
    return out

def cosh(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the hyperbolic cosine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply cosh to.
    
    Returns:
    Tensor: A tensor with the cosh function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.cosh(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = CoshBackward(tensor)
    return out



def cos(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the cosine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply cosine to.
    
    Returns:
    Tensor: A tensor with the cosine function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.cos(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = CosBackward(tensor)
    return out

def sin(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the sine applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply sine to.
    
    Returns:
    Tensor: A tensor with the sine function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sin(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SinBackward(tensor)
    return out


def tan(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the tangent applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply tangent to.
    
    Returns:
    Tensor: A tensor with the tangent function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.tan(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = TanBackward(tensor)
    return out

def sigmoid(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the sigmoid function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply sigmoid to.
    
    Returns:
    Tensor: A tensor with the sigmoid function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = 1 / (1 + np.exp(-tensor.value))
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SigmoidBackward(tensor)
    return out

def relu(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the ReLU function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply ReLU to.
    
    Returns:
    Tensor: A tensor with the ReLU function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.maximum(0, tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = ReLuBackward(tensor)
    return out

def step(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the step function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the step function to.
    
    Returns:
    Tensor: A tensor with the step function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.heaviside(tensor.value, 1)
    return gorch.Tensor(value, requires_grad=tensor.requires_grad)

def maximum(input: 'Tensor', other: 'Tensor') -> 'Tensor':
    """
    Takes two tensor objects and returns a tensor with the element-wise maximum.
    
    Args:
    input (Tensor): The first input tensor.
    other (Tensor): The second input tensor.
    
    Returns:
    Tensor: A tensor with the element-wise maximum.
    """
    if not isinstance(input, gorch.Tensor) or not isinstance(other, gorch.Tensor):
        raise ValueError("Inputs must be Tensors")
    
    value = np.maximum(input.value, other.value)
    out = gorch.Tensor(value, requires_grad=input.requires_grad or other.requires_grad)
    if out.requires_grad:
        out._grad_fn = MaximumBackward(input, other)
    return out

def minimum(input: 'Tensor', other: 'Tensor') -> 'Tensor':
    """
    Takes two tensor objects and returns a tensor with the element-wise minimum.
    
    Args:
    input (Tensor): The first input tensor.
    other (Tensor): The second input tensor.
    
    Returns:
    Tensor: A tensor with the element-wise minimum.
    """
    if not isinstance(input, gorch.Tensor) or not isinstance(other, gorch.Tensor):
        raise ValueError("Inputs must be Tensors")
    
    value = np.minimum(input.value, other.value)
    out = gorch.Tensor(value, requires_grad=input.requires_grad or other.requires_grad)
    if out.requires_grad:
        out._grad_fn = MinimumBackward(input, other)
    return out

def abs(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the absolute value applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the absolute value to.
    
    Returns:
    Tensor: A tensor with the absolute value applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.abs(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = AbsBackward(tensor)
    return out

def sqrt(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the square root applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the square root to.
    
    Returns:
    Tensor: A tensor with the square root applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.sqrt(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = SqrtBackward(tensor)
    return out

def log(tensor: 'Tensor') -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the natural logarithm applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply the natural logarithm to.
    
    Returns:
    Tensor: A tensor with the natural logarithm applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.log(tensor.value)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = LogBackward(tensor)
    return out


def leakyRelu(tensor: 'Tensor', alpha: float = 0.01) -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the Leaky ReLU function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply Leaky ReLU to.
    alpha (float, optional): The slope of the negative part of the function.
    
    Returns:
    Tensor: A tensor with the Leaky ReLU function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    out = gorch.maximum(alpha * tensor, tensor)
    return out

def softmax(tensor: 'Tensor', axis=None) -> 'Tensor':
    """
    Takes a tensor object and returns a tensor with the softmax function applied element-wise.
    
    Args:
    tensor (Tensor): The input tensor to apply softmax to.
    axis (int, optional): The axis along which the softmax will be computed. The default is the last axis.
    
    Returns:
    Tensor: A tensor with the softmax function applied element-wise.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    out = tensor.exp()
    out = out / out.sum(axis=axis, keepdims=True)
    return out

def max(tensor: 'Tensor', axis=None, keepdims=False) -> 'Tensor':
    """
    Takes a tensor object and returns the maximum of its elements along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor whose elements' maximum is to be computed.
    axis (int or tuple of ints, optional): Axis or axes along which a maximum is performed. 
                                            The default, axis=None, will compute the maximum of all the elements of the input tensor.
    keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
    
    Returns:
    Tensor: A tensor containing the maximum of the elements along the specified axis.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.max(tensor.value, axis=axis, keepdims=keepdims)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = MaxBackward(tensor, axis, keepdims)
    return out

def min(tensor: 'Tensor', axis=None, keepdims=False) -> 'Tensor':
    """
    Takes a tensor object and returns the minimum of its elements along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor whose elements' minimum is to be computed.
    axis (int or tuple of ints, optional): Axis or axes along which a minimum is performed. 
                                            The default, axis=None, will compute the minimum of all the elements of the input tensor.
    keepdims (bool, optional): If True, the axes which are reduced are left in the result as dimensions with size one.
    
    Returns:
    Tensor: A tensor containing the minimum of the elements along the specified axis.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.min(tensor.value, axis=axis, keepdims=keepdims)
    out = gorch.Tensor(value, requires_grad=tensor.requires_grad)
    if tensor.requires_grad:
        out._grad_fn = MinBackward(tensor, axis, keepdims)
    return out

def argmax(tensor: 'Tensor', axis=None) -> 'Tensor':
    """
    Takes a tensor object and returns the indices of the maximum values along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor to find the indices of the maximum values.
    axis (int, optional): Axis along which to find the maximum values. The default is to flatten the tensor.
    
    Returns:
    Tensor: A tensor containing the indices of the maximum values along the specified axis.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.argmax(tensor.value, axis=axis)
    return gorch.Tensor(value, requires_grad=False)

def argmin(tensor: 'Tensor', axis=None) -> 'Tensor':
    """
    Takes a tensor object and returns the indices of the minimum values along the specified axis.
    
    Args:
    tensor (Tensor): The input tensor to find the indices of the minimum values.
    axis (int, optional): Axis along which to find the minimum values. The default is to flatten the tensor.
    
    Returns:
    Tensor: A tensor containing the indices of the minimum values along the specified axis.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input must be a Tensor")
    
    value = np.argmin(tensor.value, axis=axis)
    return gorch.Tensor(value, requires_grad=False)