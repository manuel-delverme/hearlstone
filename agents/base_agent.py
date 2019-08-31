from abc import ABC, abstractmethod
from typing import Dict, Text, Any

import numpy as np

import specs


class Bot(ABC):
  @abstractmethod
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    raise NotImplemented

  def choose(self, observation: np.ndarray, info: specs.Info, *args, **kwargs):
    assert specs.check_info_spec(info)
    assert len(info['possible_actions'].shape) == 1
    return self._choose(observation, info)


def load_model(self, model_path=None):
  raise NotImplemented
