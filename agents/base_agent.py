from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def choose(self, observation, possible_actions) -> int:
        raise NotImplemented
