from abc import ABC, abstractmethod
from typing import Dict, Text, Any, Callable, Optional

import numpy as np

import specs


class Agent(ABC):
  @abstractmethod
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    raise NotImplemented

  def choose(self, observation: np.ndarray, info: specs.Info):
    # assert specs.check_info_spec(info)
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
    assert set(info.keys()) == {*specs.INFO_KEYS, 'original_info'} or set(info.keys()) == {*specs.INFO_KEYS,
                                                                                           *specs.OPTIONAL_INFO_KEYS,
                                                                                           'original_info'}
    assert isinstance(info['original_info'], dict)
    assert isinstance(info['original_info']['possible_actions'], tuple)
    assert info['original_info']['possible_actions'][0].card is None
    # import environments.simulator
    # assert isinstance(info['original_info']['possible_actions'][0], environments.simulator.HSsimulation.Action)

    assert specs.check_possible_actions(67, info['possible_actions'])
    return self._choose(observation, info)


def load_model(self, model_path=None):
  raise NotImplemented
