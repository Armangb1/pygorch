from abc import ABC
import pickle
import inspect

from ..parameter import Parameter


class Module(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.training = True
        self._parameters = {}

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True
        for module in self.children():
            module.train()
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        self.training = False
        for module in self.children():
            module.train()
        for param in self.parameters():
            param.requires_grad = False
        

    def parameters(self):
        for name, value in inspect.getmembers(self):
            if name == "_parameters":
                for _, dvalue in value.items():
                    yield dvalue
            elif isinstance(value, Module):
                yield from value.parameters()

    def add_parameter(self, name, param):
        if not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter, but got {type(param).__name__}")
        self._parameters[name] = param

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def state_dict(self):
        return {name: param.data for name, param in self._parameters.items()}

    def load_state_dict(self, state_dict):
        for name, param in state_dict.items():
            if name in self._parameters:
                self._parameters[name].data = param
            else:
                raise KeyError(f"Unexpected key {name} in state_dict")

    def children(self):
        res = []
        for name, value in inspect.getmembers(self):
            if isinstance(value, Module):
                res+= [value]
        return res
                

    
    