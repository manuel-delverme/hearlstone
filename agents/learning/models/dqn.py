from agents.learning.models import base_q_net
import torch.nn as nn


class DQN(base_q_net.base_QN):
  def __init__(self, num_inputs, num_actions):
    super(DQN, self).__init__(num_inputs, num_actions)
    self.layers = None

  def build_network(self):
    self.layers = nn.Sequential(
      nn.Linear(self.num_inputs + self.num_actions, 128),
      nn.ReLU(),
      # nn.Linear(128, 128),
      # nn.ReLU(),

      # 1 is the Q(s, a) value
      nn.Linear(128, 1),
      # nn.Linear(self.num_inputs + self.num_actions, 1),
      # nn.Tanh()
    )

  def forward(self, x):
    return self.layers(x)
