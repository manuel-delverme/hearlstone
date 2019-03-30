from typing import NewType
from typing import Text, Union, Dict

import numpy as np
import torch

Info = NewType('Info', Dict[Text, Union[np.ndarray, Text]])

INFO_KEYS = {'action_history', 'observation', 'possible_actions', 'reward'}
# OPTIONAL_INFO_KEYS = {'episode', 'game_statistics', 'end_episode_info'}
OPTIONAL_INFO_KEYS = {'end_episode_info'}


def check_info_spec(info: Info):
  if INFO_KEYS != set(info.keys()):
    assert set(info.keys()).difference(INFO_KEYS) == OPTIONAL_INFO_KEYS, set(info.keys())

  assert isinstance(info['action_history'], str)
  assert isinstance(info['observation'], np.ndarray)
  assert isinstance(info['possible_actions'], np.ndarray)
  assert isinstance(info['reward'], np.ndarray)

  assert info['observation'].dtype in (np.float, np.int64)
  assert info['possible_actions'].dtype in (np.float32,)
  assert info['reward'].dtype in (np.float32,)

  assert 'end_episode_info' not in info or tuple(info['end_episode_info'].keys()) == ('reward',)
  return True


def check_observation(num_inputs, observation):
  assert observation.dtype == torch.float32
  assert len(observation.size()) == 2  # batch_size, num_inputs
  assert observation.size(1) == num_inputs
  return True


def check_action(action):
  assert action.dtype == torch.int64
  assert len(action.size()) == 2  # batch_size, 1
  assert action.size(1) == 1
  return True


def check_possible_actions(num_possible_actions, possible_actions):
  assert possible_actions.dtype == torch.float32
  assert possible_actions.shape[1:] == (num_possible_actions,)
  return True


def check_positive_type(value, type_, strict=True):
  assert isinstance(value, type_)
  if strict:
    assert value > 0
  else:
    assert value > -1
  return True
