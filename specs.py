from typing import Text, Union, Dict
from typing import NewType

import numpy as np

Info = NewType('Info', Dict[Text, Union[np.ndarray, Text]])

INFO_KEYS = ['action_history', 'observation', 'possible_actions', 'reward']
OPTIONAL_INFO_KEYS = ['episode', 'game_statistics']


def check_info_spec(info: Info):
  assert sorted(info.keys()) == INFO_KEYS or sorted(info.keys()) == sorted(INFO_KEYS + OPTIONAL_INFO_KEYS)

  assert isinstance(info['action_history'], str)
  assert isinstance(info['observation'], np.ndarray)
  assert isinstance(info['possible_actions'], np.ndarray)
  assert isinstance(info['reward'], np.ndarray)

  assert info['observation'].dtype in (np.float, np.int64)
  assert info['possible_actions'].dtype in (np.float32,)
  assert info['reward'].dtype in (np.float32,)


def check_inputs(observation_shape, inputs):
  assert len(inputs.size()) == 2
  assert inputs.size()[1:] == observation_shape


def check_possible_actions(num_possible_actions, possible_actions):
  assert possible_actions.shape[1:] == (num_possible_actions,)
