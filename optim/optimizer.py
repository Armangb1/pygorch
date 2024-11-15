from abc import ABC


class Optimizer(ABC):

    def __init__(self):
        pass

    def step(self):
        raise NotImplementedError

    def zero_grad():
        pass