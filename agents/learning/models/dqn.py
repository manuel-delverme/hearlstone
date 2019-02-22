from agents.learning.models import noisy_networks
import config
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import ndarray
from torch import Tensor
from typing import Union


class DQN(nn.Module):
  def __init__(self, num_inputs: int, num_actions: int) -> None:
    self.num_inputs = num_inputs
    self.num_actions = num_actions
    self.use_cuda = config.use_gpu

    super(DQN, self).__init__()

  def build_network(self) -> None:
    self.linear1 = nn.Linear(self.num_inputs, 64)
    # self.linear2 = nn.Linear(32, 64)

    # self.value1 = nn.Linear(64, 64)
    self.value_out = nn.Linear(64, 1)

    # self.advantage1 = nn.Linear(64, 64)
    self.advantage_out = nn.Linear(64, self.num_actions)

    if self.use_cuda:
      self.cuda()

  def forward(self, x: Union[ndarray, Tensor]) -> Tensor:
    x = torch.Tensor(x)
    if self.use_cuda:
      x = x.cuda()

    x = F.relu(self.linear1(x))
    # x = F.relu(self.linear2(x))

    value = x
    # value = F.relu(self.value1(x))
    value = self.value_out(value)

    advantage = x
    # advantage = F.relu(self.advantage1(x))
    advantage = self.advantage_out(advantage)

    q_val = value + advantage - advantage.mean()

    if self.use_cuda:
      q_val = q_val.cpu()
    return q_val

  def get_value(self, x: Union[ndarray, Tensor]) -> Tensor:
    x = torch.Tensor(x)
    if self.use_cuda:
      x = x.cuda()

    x = F.relu(self.linear1(x))
    # x = F.relu(self.linear2(x))

    value = x
    # value = F.relu(self.value1(x))
    value = self.value_out(value)

    if self.use_cuda:
      value = value.cpu()
    return value
