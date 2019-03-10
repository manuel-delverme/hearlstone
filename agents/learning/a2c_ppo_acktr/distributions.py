import torch
import torch.nn as nn

import hs_config
from agents.learning.a2c_ppo_acktr.utils import init

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0),
                                                                                                -1).sum(-1).unsqueeze(
  -1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
  def __init__(self, num_inputs, num_outputs):
    super(Categorical, self).__init__()

    init_ = lambda m: init(m,
                           nn.init.orthogonal_,
                           lambda x: nn.init.constant_(x, 0),
                           gain=0.01)

    self.linear = init_(nn.Linear(num_inputs, num_outputs))

  def forward(self, x, possible_actions):
    x = self.linear(x)
    x -= (1 - possible_actions) * hs_config.BIG_NUMBER
    return FixedCategorical(logits=x)
