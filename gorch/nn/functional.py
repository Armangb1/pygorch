import gorch
import numpy as np

__all__ = ['one_hot', 'jacobian', 'vectorize_parameters', 'devectorize_parameters']   
def one_hot(tensor: gorch.Tensor, num_classes = -1):
    """
    One-hot encode labels.
    """
    if not isinstance(tensor, gorch.Tensor):
        raise ValueError("Input should be a tensor")
    if len(tensor.shape) > 1:
        if tensor.shape[0] == 1:
            tensor = tensor.transpose()
        elif tensor.shape[1] == 1:
            pass
        else:
            raise ValueError("One-hot encoding is only applicable to 1D tensors")
    
    if num_classes == -1:
        num_classes = np.max(tensor.value)+1
    else:
        num_classes = num_classes

    one_hot = np.zeros((tensor.value.shape[0], num_classes))
    one_hot[np.arange(one_hot.shape[0]), tensor.value.reshape(-1).astype(np.int8)-1] = 1
    return gorch.Tensor(one_hot)


def jacobian(model, input: gorch.Tensor):
    """
    Calculate the Jacobian of the model with respect to the input tensor.
    """
    if not isinstance(input, gorch.Tensor):
        raise ValueError("Input should be a tensor")
    if not isinstance(model, gorch.nn.Module):
        raise ValueError("Model should be a module")
    
    y_pred = model(input)
    outDim = int(np.array(y_pred.shape).prod())
    numParam = 0
    numOut = y_pred.shape[1]
    for p in model.parameters():
        numParam += int(np.array(p.shape).prod())
    jac = np.zeros((outDim, numParam))
    for batchIdx in range(y_pred.shape[0]):
        y_i = model(input[[batchIdx],:])
        for no in range(numOut):
            gradient = gorch.zeros(numOut,numOut)
            gradient[no, no] = 1
            zero_grad(model)
            y_i.backward(gradient)
            _, valgrad = vectorize_parameters(model)
            jac[(batchIdx)*numOut+no,:] = valgrad
    return jac

def vectorize_parameters(model):
    """
    Vectorize the parameters of the model.
    """
    if not isinstance(model, gorch.nn.Module):
        raise ValueError("Model should be a module")
    val =  np.concatenate([p.value.reshape(-1) for p in model.parameters()])
    valgrad = np.concatenate([p.grad.value.reshape(-1) for p in model.parameters()])
    return val, valgrad

def devectorize_parameters(model, val):
    """
    Devectorize the parameters of the model.
    """
    if not isinstance(model, gorch.nn.Module):
        raise ValueError("Model should be a module")
    idx = 0
    for p in model.parameters():
        nextIdx = 0
        for i in p.shape:
            nextIdx += i
        p.value = val[idx:nextIdx].reshape(p.value.shape)
        idx = nextIdx+1
def zero_grad(model):
    """
    Zero the gradients of the model.
    """
    if not isinstance(model, gorch.nn.Module):
        raise ValueError("Model should be a module")
    for p in model.parameters():
        p.grad = None
        p._grad_fn = None