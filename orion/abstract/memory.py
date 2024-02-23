from abc import ABC, abstractmethod
from typing import Any

from attr import define


@define
class EpisodicMemory(ABC):
    """
    Base memory that can store experiences
    """

    positive_memory: Any
    negative_memory: Any

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def delete(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def retrieve(self, *args, **kwargs):
        pass
