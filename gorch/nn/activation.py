from .modules.module import Module
import gorch
class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, **kwargs):
        return gorch.sigmoid(inputs)
        
class Relu(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, **kwargs):
        return gorch.relu(inputs)
        
class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, **kwargs):
        return gorch.tanh(inputs)
      
class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, **kwargs):
        return gorch.softmax(inputs)
    
class LeakyRelu(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, **kwargs):
        return gorch.leakyRelu(inputs)