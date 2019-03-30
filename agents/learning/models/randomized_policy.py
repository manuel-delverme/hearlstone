import numpy as np
import torch
import torch.nn as nn

import hs_config
import specs
from shared.utils import init


class ActorCritic(nn.Module):
  def __init__(self, num_inputs: int, num_actions: int):
    super(ActorCritic, self).__init__()
    assert num_inputs > 0
    assert num_actions > 0

    self.num_inputs = num_inputs
    self.num_possible_actions = num_actions

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
    self.actor = nn.Sequential(
      init_(nn.Linear(self.num_inputs, hs_config.PPOAgent.hidden_size)),
      nn.SELU(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.SELU(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.SELU(),
    )
    self.critic = nn.Sequential(
      init_(nn.Linear(self.num_inputs, hs_config.PPOAgent.hidden_size)),
      nn.SELU(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.SELU(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.SELU(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, 1)),
    )

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
    self.actor_logits = init_(nn.Linear(hs_config.PPOAgent.hidden_size, self.num_possible_actions))

    self.train()

  def forward(self, observations: torch.FloatTensor, possible_actions: torch.FloatTensor,
    deterministic: bool = False) -> (
    torch.FloatTensor, torch.LongTensor, torch.FloatTensor):

    assert specs.check_observation(self.num_inputs, observations)
    assert specs.check_possible_actions(self.num_possible_actions, possible_actions)
    assert observations.size(0) == possible_actions.size(0)
    assert isinstance(deterministic, bool)

    action_distribution, value = self.actor_critic(observations, possible_actions)

    if deterministic:
      action = action_distribution.probs.argmax(dim=-1, keepdim=True)
    else:
      action = action_distribution.sample().unsqueeze(-1)

    action_log_probs = self.action_log_prob(action, action_distribution)

    assert value.size(1) == 1
    assert specs.check_action(action)
    assert action_log_probs.size(1) == 1
    assert value.size(0) == action.size(0) == action_log_probs.size(0)

    return value, action, action_log_probs

  def action_log_prob(self, action, action_distribution):
    assert action.size(1) == 1

    action_log_probs = action_distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    assert action_log_probs.size(1) == 1
    return action_log_probs

  def actor_critic(self, inputs, possible_actions):
    value = self.critic(inputs)
    actor_features = self.actor(inputs)

    logits = self.actor_logits(actor_features)
    action_distribution = self._get_action_distribution(possible_actions, logits)
    return action_distribution, value

  def evaluate_actions(self, observations: torch.FloatTensor, action: torch.LongTensor,
    possible_actions: torch.FloatTensor) -> (
    torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):

    assert specs.check_observation(self.num_inputs, observations)
    assert specs.check_possible_actions(self.num_possible_actions, possible_actions)
    assert specs.check_action(action)
    assert action.size(0) == observations.size(0) == possible_actions.size(0)

    action_distribution, value = self.actor_critic(observations, possible_actions)
    action_log_probs = self.action_log_prob(action, action_distribution)
    dist_entropy = action_distribution.entropy().mean()

    assert value.size(1) == 1
    assert action_log_probs.size(1) == 1
    assert not dist_entropy.size()
    assert value.size(0) == action_log_probs.size(0)

    return value, action_log_probs, dist_entropy

  @staticmethod
  def _get_action_distribution(possible_actions, logits):
    logits -= ((1 - possible_actions) * hs_config.BIG_NUMBER).float()
    return torch.distributions.Categorical(logits=logits)
