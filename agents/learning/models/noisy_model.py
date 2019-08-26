#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional
import math
from shared.utils import init


class FactorisedNoisyLinear(nn.Module):
  """Factorised Gaussian NoisyNet"""

  def __init__(self, in_features, out_features, sigma_0=0.5, bias=True):
    super(FactorisedNoisyLinear, self).__init__()

    self.in_features = in_features
    self.out_features = out_features
    self.sigma = sigma_0 / math.sqrt(in_features)

    self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    self.weight = self.linear.weight
    self.bias = self.linear.bias

    self.factorised_linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    self.in_noise = torch.FloatTensor(in_features)
    self.out_noise = torch.FloatTensor(out_features)
    self.noise_matrix = None

    self.reset()

  def reset(self):
    self.sample_noise()
    self.reset_linear()
    self.reset_noise()

  def sample_noise(self):
    self.in_noise.normal_(0, self.sigma)
    self.out_noise.normal_(0, self.sigma)
    self.noise_matrix = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

  def reset_linear(self):
    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.1)
    self.linear.apply(init_)

  def reset_noise(self):
    std = 1. / math.sqrt(self.linear.weight.size(1))
    init_ = lambda m: init(m, nn.init.uniform_, lambda x: nn.init.uniform_, a=-std, b=std)
    self.factorised_linear.apply(init_)

  def forward(self, x, deterministic=False):
    x = self.linear(x)
    if deterministic:
      noisy_weight = self.factorised_linear.weight * self.noise_matrix
      noisy_bias = self.factorised_linear.bias * self.out_noise
      x_noisy = nn.functional.linear(x, noisy_weight, noisy_bias)
      return x + x_noisy
    return x

  def __repr__(self):
    return f"{self.__class__.__name__}:(deterministic={self.deterministic}, in_features={self.in_features},out_features={self.out_features})"

# if __name__ == "__main__":
#   net = FactorisedNoisyLinear(10, 10)
#   print(net)

# # Noisy linear layer with independent Gaussian noise
# class NoisyLinear(nn.Linear):
#   def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
#     super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
#     # µ^w and µ^b reuse self.weight and self.bias
#     self.sigma_init = sigma_init
#     self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
#     self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
#     self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
#     self.register_buffer('epsilon_bias', torch.zeros(out_features))
#     self.reset_parameters()
#
#   def reset_parameters(self):
#     if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
#       init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#       init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#       init.constant(self.sigma_weight, self.sigma_init)
#       init.constant(self.sigma_bias, self.sigma_init)
#
#   def forward(self, input):
#     return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight),
#                     self.bias + self.sigma_bias * Variable(self.epsilon_bias))
#
#   def sample_noise(self):
#     self.epsilon_weight = torch.randn(self.out_features, self.in_features)
#     self.epsilon_bias = torch.randn(self.out_features)
#
#   def remove_noise(self):
#     self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
#     self.epsilon_bias = torch.zeros(self.out_features)
