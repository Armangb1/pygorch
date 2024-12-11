from .module import Module
from gorch import Tensor
import gorch

class _Loss(Module):
    def __init__(self, reduce = None, reduction:str="mean") -> None:
        super().__init__()
        if reduce is not None:
            reduction = reduce
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction should be 'mean' or 'sum', got {reduction}")
        if reduction == 'mean':
            self.reduction = gorch.mean
        else:
            self.reduction = gorch.sum

class L1Loss(_Loss):
    def __init__(self, reduce = None, reduction:str = "mean") -> None:
        super().__init__(reduce, reduction)
    
    def forward(self, input:Tensor, target:Tensor):
        e = target-input
        L = gorch.sum(gorch.abs(e),axis=tuple(range(1,len(e.shape)-1)))
        return self.reduction(L)

class MSELoss(_Loss):
    def __init__(self, reduce = None, reduction:str="mean") -> None:
        super().__init__(reduce, reduction)

    def forward(self, input:Tensor, target:Tensor):
        e = target-input
        L = gorch.sum(e**2,axis=tuple(range(1,len(e.shape)-1)))
        return self.reduction(L)
    
class BCELoss(_Loss):
    def __init__(self, reduce = None, reduction:str="mean") -> None:
        super().__init__(reduce, reduction)
    
    def forward(self, input:Tensor, target:Tensor):
        e = input
        L = -gorch.sum(target*gorch.log(e)+(1-target)*gorch.log(1-e),axis=tuple(range(1,len(e.shape)-1)))
        return self.reduction(L)
    
class CrossEntropyLoss(_Loss):
    def __init__(self, reduce = None, reduction:str="mean", eps = 1e-6) -> None:
        super().__init__(reduce, reduction)
        self.eps = eps
    def forward(self, input:Tensor, target:Tensor):
        e = input
        L = -gorch.sum(target*gorch.log(e+self.eps),axis=tuple(range(1,len(e.shape)-1)))
        return self.reduction(L)