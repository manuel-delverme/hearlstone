from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def choose(self, state, possible_actions):
        raise NotImplemented
