import torch
import torch.nn as nn
from agents.learning import base_q_net


class DuelingDQN(base_q_net.BaseQLearner):
  def __init__(self, num_inputs, num_actions):
    super().__init__(num_inputs, num_actions)
    self.feature = None
    self.advantage = None
    self.value = None

  def build_network(self):
    hidden_size = 128
    self.feature = nn.Sequential(
      nn.Linear(self.num_inputs, hidden_size),
      nn.ReLU()
    )
    self.advantage = nn.Sequential(
      nn.Linear(hidden_size + self.num_actions, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1)
    )

    self.value = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1)
    )

  def forward(self, x):
    obs = x[:, :self.num_inputs]
    action = x[:, self.num_inputs:]
    h = self.feature(obs)
    value = self.value(h)

    h_advantage = torch.cat((h, action), 1)
    advantage = self.advantage(h_advantage)
    adv_mean = advantage.mean()
    q_value = value + advantage - adv_mean
    raise NotImplementedError('advantage mean is not well defined if not all actions are available, fix')
    return q_value
