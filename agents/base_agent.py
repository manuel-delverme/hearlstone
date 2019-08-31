from abc import ABC, abstractmethod

import numpy as np

import specs


class Bot(ABC):
  @abstractmethod
  def choose_greedy(self, observation: np.ndarray, info: specs.Info):
    raise NotImplemented

  def choose(self, observation: np.ndarray, info: specs.Info):
    assert specs.check_info_spec(info)
    assert len(info['possible_actions'].shape) == 1
    return self.choose_greedy(observation, info)
