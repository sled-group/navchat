from abc import ABC, abstractmethod


class UserSimulator(ABC):
    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
