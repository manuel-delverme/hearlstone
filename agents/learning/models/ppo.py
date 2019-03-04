import torch.distributions
import torch.nn as nn
import hs_config
import torch


def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, mean=0., std=0.1)
    nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
  def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
    super(ActorCritic, self).__init__()

    self.fc1 = nn.Linear(num_inputs, hidden_size)
    # self.actor_fc1 = nn.Linear(num_inputs, hidden_size)

    self.critic_fc2 = nn.Linear(hidden_size, 1)
    self.actor_fc2 = nn.Linear(hidden_size, num_outputs)
    self.apply(init_weights)

    self.use_cuda = hs_config.DQNAgent.use_gpu
    if self.use_cuda:
      self.cuda()
    self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

  def forward(self, x):
    x = torch.Tensor(x)
    if self.use_cuda:
      x = x.cuda()

    h = self.fc1(x)

    value = self.critic_fc2(h)
    mu = self.actor_fc2(h)

    if self.use_cuda:
      value = value.cpu()
      mu = mu.cpu()

    std = self.log_std.exp().expand_as(mu)
    return mu, std, value
