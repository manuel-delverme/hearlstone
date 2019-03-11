from typing import Sequence

import gym.spaces
import numpy as np
import torch
import torch.nn as nn

import hs_config
from agents.learning.a2c_ppo_acktr.utils import init


class Policy(nn.Module):
  def __init__(self, obs_shape: Sequence, action_space: gym.spaces.Discrete):
    assert len(obs_shape) == 1
    assert action_space.shape == tuple()

    super(Policy, self).__init__()

    self.observation_shape = obs_shape
    self.action_shape = (1,)
    self.num_possible_actions = action_space.n

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
    self.actor = nn.Sequential(
      init_(nn.Linear(self.observation_shape[0], hs_config.PPOAgent.hidden_size)),
      nn.Tanh(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.Tanh()
    )
    self.critic = nn.Sequential(
      init_(nn.Linear(self.observation_shape[0], hs_config.PPOAgent.hidden_size)),
      nn.Tanh(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.Tanh(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, 1)),
    )

    self.train()

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
    self.actor_logits = init_(nn.Linear(hs_config.PPOAgent.hidden_size, action_space.n))

  def forward(self, inputs: torch.FloatTensor, possible_actions: torch.FloatTensor, deterministic: bool = False) -> (
    torch.FloatTensor, torch.LongTensor, torch.FloatTensor):

    self._check_inputs_and_possible_actions(inputs, possible_actions)

    value = self.critic(inputs)
    actor_features = self.actor(inputs)

    logits = self.actor_logits(actor_features)
    action_distribution = self._get_action_distribution(possible_actions, logits)

    if deterministic:
      action = action_distribution.probs.argmax(dim=-1, keepdim=True)  # dist.mode()
    else:
      action = action_distribution.sample().unsqueeze(-1)

    # TODO: remove this after multiprocess is tested
    # action_log_probs = action_distribution.log_prob(action.squeeze(-1))
    # action_log_probs = action_log_probs.view(action.size(0), -1).sum(-1).unsqueeze(-1)  # TODO: simplify
    action_log_probs = action_distribution.log_prob(action)

    assert value.shape == (inputs.size(0), 1)
    assert action.shape == (inputs.size(0), 1)
    assert action_log_probs.shape == (inputs.size(0), 1)
    return value, action, action_log_probs

  def evaluate_actions(self, inputs: torch.FloatTensor, action: torch.LongTensor, possible_actions: torch.FloatTensor) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):

    self._check_inputs_and_possible_actions(inputs, possible_actions)
    assert action.shape == (inputs.size(0), 1)

    value = self.critic(inputs)
    actor_features = self.actor(inputs)

    logits = self.actor_logits(actor_features)
    action_distribution = self._get_action_distribution(possible_actions, logits)

    action_log_probs = action_distribution.log_prob(action)
    # action_log_probs = dist.log_prob(action.squeeze(-1))
    # action_log_probs = action_log_probs.view(action.size(0), -1).sum(-1).unsqueeze(-1)  # TODO: simplify
    dist_entropy = action_distribution.entropy().mean()

    assert value.shape == (inputs.size(0), 1)
    assert action_log_probs.shape == (inputs.size(0), 1)
    assert len(dist_entropy.size()) == 0

    return value, action_log_probs, dist_entropy

  def _check_inputs_and_possible_actions(self, inputs, possible_actions):
    assert len(inputs.size()) == 2
    assert inputs.size()[1:] == self.observation_shape
    assert possible_actions.shape[1:] == (self.num_possible_actions,)
    assert inputs.size(0) == possible_actions.size(0)

  @staticmethod
  def _get_action_distribution(possible_actions, logits):
    logits -= (1 - possible_actions) * hs_config.BIG_NUMBER
    return torch.distributions.Categorical(logits=logits)
