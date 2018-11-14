import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn

import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args,
                                                     **kwargs).cuda() if USE_CUDA else autograd.Variable(
  *args, **kwargs)


class base_QN(nn.Module):
  def __init__(self, num_inputs, num_actions):
    super(base_QN, self).__init__()
    self.num_inputs = num_inputs
    self.num_actions = num_actions

  def build_network(self):
    raise NotImplementedError

  def forward(self, x):
    raise NotImplementedError

  def act(self, state: np.array, possible_actions: List[Tuple[int, int]],
    epsilon: float):
    assert isinstance(state, np.ndarray)
    assert isinstance(possible_actions, list)
    assert isinstance(possible_actions[0], tuple)
    assert isinstance(possible_actions[0][0], int)
    assert isinstance(epsilon, float)

    if random.random() > epsilon:
      network_inputs = []
      for possible_action in possible_actions:
        network_input = np.append(state, possible_action)
        network_inputs.append(network_input)

      network_inputs = np.array(network_inputs)
      network_input = Variable(torch.FloatTensor(network_inputs).unsqueeze(0),
                               volatile=True)
      q_values = self.forward(network_input).cpu().data.numpy()

      best_action = np.argmax(q_values)
      action = possible_actions[best_action]
    else:
      action, = random.sample(possible_actions, 1)
    return action
