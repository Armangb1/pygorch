from .tensor import Tensor
import numpy as np

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
    return Tensor(value, required_grad=tensor.required_grad)


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
    return Tensor(value, required_grad=tensor.required_grad)


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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)



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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)



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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)


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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)

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
    return Tensor(value, required_grad=tensor.required_grad)