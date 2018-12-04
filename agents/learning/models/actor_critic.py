from agents.learning.models import noisy_networks
import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2C(nn.Module):
  def __init__(self, num_inputs, num_actions):
    self.num_inputs = num_inputs
    self.num_actions = num_actions
    self.use_cuda = config.use_gpu
    super().__init__()

  def build_network(self):
    self.critic = nn.Sequential(
      nn.Linear(self.num_inputs, 256),
      nn.ReLU(),
      nn.Linear(256, 1)
    )

    self.actor = nn.Sequential(
      nn.Linear(self.num_inputs, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_actions),
    )

    if self.use_cuda:
      self.cuda()

  def forward(self, x, possible_actions):
    x = torch.Tensor(x)
    possible_actions = torch.Tensor(possible_actions)
    if self.use_cuda:
      x = x.cuda()
      possible_actions = possible_actions.cuda()

    value = self.critic(x)
    logits = self.actor(x)

    all_probs = F.softmax(logits, dim=1)
    possible_logits = all_probs * possible_actions
    probs = F.normalize(possible_logits, dim=1, p=1)

    dist = torch.distributions.Categorical(probs=probs)

    return dist, value
