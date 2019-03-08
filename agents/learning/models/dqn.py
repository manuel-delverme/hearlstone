from agents.learning.models import noisy_networks
import hs_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
  def __init__(self, num_inputs, num_actions):
    self.num_inputs = num_inputs
    self.num_actions = num_actions
    self.use_cuda = hs_config.use_gpu
    self.num_atoms = 51
    self.Vmin = -10
    self.Vmax = 10
    super(DQN, self).__init__()

  def build_network(self):
    self.linear1 = nn.Linear(self.num_inputs, 32)
    self.linear2 = nn.Linear(32, 64)

    self.value_nosiy_fc3 = noisy_networks.NoisyLinear(256, self.num_actions)
    self.noisy_value1 = noisy_networks.NoisyLinear(64, 64)
    self.noisy_value2 = noisy_networks.NoisyLinear(64, self.num_atoms)

    self.noisy_advantage1 = noisy_networks.NoisyLinear(64, 64)
    self.noisy_advantage2 = noisy_networks.NoisyLinear(64, self.num_atoms * self.num_actions)

    if self.use_cuda:
      self.cuda()

  def value_network(self, x):
    h = self.value_fc1(x)
    h = F.leaky_relu(h)
    h = self.value_fc2(h)
    h = F.leaky_relu(h)
    h = self.value_nosiy_fc3(h)
    return h

  def forward(self, x):
    x = torch.Tensor(x)
    if self.use_cuda:
      x = x.cuda()

    batch_size = x.size(0)

    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))

    value = F.relu(self.noisy_value1(x))
    value = self.noisy_value2(value)

    advantage = F.relu(self.noisy_advantage1(x))
    advantage = self.noisy_advantage2(advantage)

    value = value.view(batch_size, 1, self.num_atoms)
    advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

    all_q_val = value + advantage - advantage.mean(1, keepdim=True)
    q_val = F.softmax(all_q_val.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

    if self.use_cuda:
      q_val = q_val.cpu()
    return q_val

  def reset_noise(self):
    self.noisy_value1.reset_noise()
    self.noisy_value2.reset_noise()
    self.noisy_advantage1.reset_noise()
    self.noisy_advantage2.reset_noise()
