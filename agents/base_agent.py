from abc import ABC, abstractmethod
from environments import base_env


class Agent(ABC):
    @abstractmethod
    def choose(self, observation, possible_actions) -> base_env.BaseEnv.GameActions:
        raise NotImplemented
