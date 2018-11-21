import torch.nn as nn
import torch


class DQN(nn.Module):
  def __init__(self, num_inputs, num_actions, use_cuda):
    self.layers = None
    self.num_inputs = num_inputs
    self.num_actions = num_actions
    self.use_cuda = use_cuda

    super(DQN, self).__init__()

  def build_network(self):
    self.layers = nn.Sequential(
      nn.Linear(self.num_inputs + self.num_actions, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),

      # 1 is the Q(s, a) value
      nn.Linear(128, 1),
      # nn.Linear(self.num_inputs + self.num_actions, 1),
      # nn.Tanh()
    )
    if self.use_cuda:
      self.cuda()

  def forward(self, x):
    x = torch.FloatTensor(x)
    if self.use_cuda:
      x = x.cuda()
    retr = self.layers(x)
    if self.use_cuda:
      retr = retr.cpu()
    return retr
