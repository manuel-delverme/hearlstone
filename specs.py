from typing import NewType
from typing import Text, Union, Dict

import numpy as np
import torch

Info = NewType('Info', Dict[Text, Union[np.ndarray, Text]])

INFO_KEYS = {
  'possible_actions',
}
TERMINAL_GAME_INFO_KEYS = {
  'game_statistics',
}
BOT_INFO_KEYS = {
  'original_info',
}


def check_info_spec(info: Info):
  if INFO_KEYS != set(info.keys()):
    keys = set(info.keys())
    if keys.difference(INFO_KEYS) == TERMINAL_GAME_INFO_KEYS:
      assert isinstance(info['game_statistics'], dict)
      [float(s) for s in info['game_statistics'].values()]

    elif keys.difference(INFO_KEYS) == BOT_INFO_KEYS:

      assert isinstance(info['original_info'], dict)
      assert set(info['original_info'].keys()) == {'game_options', 'game_snapshot'}

      assert isinstance(info['original_info']['game_options'], dict)
      import environments.sabber_hs
      assert isinstance(info['original_info']['game_snapshot'], environments.sabber_hs._GameRef)

    else:
      raise AssertionError('invalid info format')

  # assert isinstance(info['action_history'], list)
  # assert isinstance(info['observation'], np.ndarray)
  assert isinstance(info['possible_actions'], np.ndarray)
  # assert isinstance(info['reward'], np.ndarray)

  # assert info['observation'].dtype in (np.float, np.int64, np.int32)
  assert info['possible_actions'].dtype in (np.float32,)
  # assert info['reward'].dtype in (np.float32,)

  assert 'end_episode_info' not in info or tuple(info['end_episode_info'].keys()) == ('reward',)
  return True


def check_observation(num_inputs, observation, check_batch=True):
  if isinstance(observation, torch.Tensor):
    if observation.dtype not in (torch.float32,):
      raise TypeError("{} is not correct".format(observation.dtype))
  elif isinstance(observation, np.ndarray):
    if observation.dtype not in (np.int32,):
      raise TypeError("{} is not correct".format(observation.dtype))
  else:
    raise TypeError

  if check_batch:
    if len(observation.shape) != 2:  # batch_size, num_inputs
      raise TypeError
    if observation.shape[1] != num_inputs:
      raise TypeError
  return True


def check_action(action):
  assert action.dtype in (torch.int64, np.int64)
  assert len(action.size()) == 2  # batch_size, 1
  assert action.size(1) == 1
  return True


def check_possible_actions(num_possible_actions, possible_actions):
  # if isinstance(possible_actions, np.ndarray):
  #   if possible_actions.dtype != np.float32:
  #     raise TypeError("{} is not correct".format(possible_actions.dtype))
  if isinstance(possible_actions, torch.Tensor):
    if possible_actions.dtype not in (torch.float32,):
      raise TypeError("{} is not correct".format(possible_actions.dtype))

  if num_possible_actions is not None:
    assert possible_actions.shape[1:] == (num_possible_actions,)
  return True


def check_positive_type(value, type_, strict=True):
  assert isinstance(value, type_)
  if strict:
    assert value > 0
  else:
    assert value > -1
  return True
