from .modules.module import Module

class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs, **kwargs):
        return inputs.sigmoid()
        
class Relu(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs, **kwargs):
        return inputs.relu()
        
class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs, **kwargs):
        return inputs.tanh()

