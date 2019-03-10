from abc import ABC, abstractmethod
from typing import Dict, Text, Any

import numpy as np


class Agent(ABC):
  @abstractmethod
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    raise NotImplemented

  def choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    import specs
    specs.check_info_spec(info)
    return self._choose(observation, info)

  def load_model(self, model_path=None):
    raise NotImplemented
