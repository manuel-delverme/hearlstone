from abc import ABC, abstractmethod
from typing import Dict, Text, Any, Callable, Optional

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

  def train(self, load_env: Callable, checkpoint_file: Optional[Text], num_updates: int, updates_offset: int) -> Text:
    raise NotImplemented

  def self_play(self, game_manger, checkpoint_file):
    raise NotImplemented


class Bot(ABC):
  @abstractmethod
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    raise NotImplemented

  def choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    import environments.simulator
    assert sorted(info.keys()) == ['original_info', 'possible_actions']
    assert isinstance(info['original_info'], dict)
    assert isinstance(info['original_info']['possible_actions'], tuple)
    assert isinstance(info['original_info']['possible_actions'][0], environments.simulator.HSsimulation.Action)

    assert isinstance(info['possible_actions'], tuple)
    assert isinstance(info['possible_actions'][0], int)
    return self._choose(observation, info)

  def load_model(self, model_path=None):
    raise NotImplemented
