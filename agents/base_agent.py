from abc import ABC


class Agent(ABC):
    def __init__(self):
        pass

    def choose(self, state, possible_actions):
        raise NotImplemented
