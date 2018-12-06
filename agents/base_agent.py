from abc import ABC, abstractmethod
import torch


class Agent(ABC):
    @abstractmethod
    def choose(self, observation, possible_actions):
        raise NotImplemented

    def load_model(self, model_path=None):
      if model_path is None:
        model_path = self.model_path
      self.network.load_state_dict(torch.load(model_path))
      print('loaded', model_path)
