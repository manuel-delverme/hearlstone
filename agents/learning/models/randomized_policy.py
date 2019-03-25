import numpy as np
import torch
import torch.nn as nn

import hs_config
import specs
from shared.utils import init


class Policy(nn.Module):
  def __init__(self, num_inputs: int, num_actions: int):
    super(Policy, self).__init__()
    assert num_inputs > 0
    assert num_actions > 0

    self.num_inputs = num_inputs
    self.num_possible_actions = num_actions

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
    self.actor = nn.Sequential(
      init_(nn.Linear(self.num_inputs, hs_config.PPOAgent.hidden_size)),
      nn.Tanh(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.Tanh()
    )
    self.critic = nn.Sequential(
      init_(nn.Linear(self.num_inputs, hs_config.PPOAgent.hidden_size)),
      nn.Tanh(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, hs_config.PPOAgent.hidden_size)),
      nn.Tanh(),
      init_(nn.Linear(hs_config.PPOAgent.hidden_size, 1)),
    )

    self.train()

    init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
    self.actor_logits = init_(nn.Linear(hs_config.PPOAgent.hidden_size, self.num_possible_actions))

  def forward(self, inputs: torch.FloatTensor, possible_actions: torch.FloatTensor, deterministic: bool = False) -> (
    torch.FloatTensor, torch.LongTensor, torch.FloatTensor):

    specs.check_observation(self.num_inputs, inputs)
    specs.check_possible_actions(self.num_possible_actions, possible_actions)
    assert inputs.size(0) == possible_actions.size(0)
    assert isinstance(deterministic, bool)

    value = self.critic(inputs).squeeze(-1)
    actor_features = self.actor(inputs)

    logits = self.actor_logits(actor_features)
    action_distribution = self._get_action_distribution(possible_actions, logits)

    if deterministic:
      action = action_distribution.probs.argmax(dim=-1)  # dist.mode()
    else:
      action = action_distribution.sample()

    # TODO: remove this after multiprocess is tested
    # action_log_probs = action_distribution.log_prob(action.squeeze(-1))
    # action_log_probs = action_log_probs.view(action.size(0), -1).sum(-1).unsqueeze(-1)  # TODO: simplify
    action_log_probs = action_distribution.log_prob(action)

    value = value.unsqueeze(-1)
    action = action.unsqueeze(-1)
    action_log_probs = action_log_probs.unsqueeze(-1)

    assert value.shape == (inputs.size(0), 1)
    assert action.shape == (inputs.size(0), 1)
    assert action_log_probs.shape == (inputs.size(0), 1)
    return value, action, action_log_probs

  def evaluate_actions(self, inputs: torch.FloatTensor, action: torch.LongTensor,
                       possible_actions: torch.FloatTensor) -> (
    torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):

    specs.check_observation(self.num_inputs, inputs)
    specs.check_possible_actions(self.num_possible_actions, possible_actions)
    assert action.size() == (inputs.size(0), 1)

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

  @staticmethod
  def _get_action_distribution(possible_actions, logits):
    logits -= ((1 - possible_actions) * hs_config.BIG_NUMBER).float()
    return torch.distributions.Categorical(logits=logits)
