from abc import ABC, abstractmethod
from typing import Dict, Text, Any, Callable, Optional

import numpy as np

import specs


class Agent(ABC):
  @abstractmethod
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    raise NotImplemented

  def choose(self, observation: np.ndarray, info: specs.Info):
    assert specs.check_info_spec(info)
    # assert specs.check_observation(C._STATE_SPACE, observation)
    return self._choose(observation, info['possible_actions'])

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

  def choose(self, observation: np.ndarray, info: specs.Info):
    assert specs.check_info_spec(info)
    assert len(info['possible_actions'].shape) == 1
    return self._choose(observation, info)


def load_model(self, model_path=None):
  raise NotImplemented
