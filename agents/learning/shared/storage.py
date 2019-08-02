from typing import List

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import hs_config
import specs


class RolloutStorage(object):
  def __init__(self, num_inputs: int, num_actions: int):
    assert specs.check_positive_type(num_inputs, int)
    assert specs.check_positive_type(num_actions, int)

    num_steps = hs_config.PPOAgent.num_steps
    num_processes = hs_config.PPOAgent.num_processes
    device = hs_config.device

    self.gamma = hs_config.PPOAgent.gamma
    self.tau = hs_config.PPOAgent.tau

    assert specs.check_positive_type(num_steps, int)
    assert specs.check_positive_type(num_processes, int)
    assert isinstance(device, torch.device)
    assert specs.check_positive_type(self.gamma, float)
    assert specs.check_positive_type(self.tau, float)

    self._observations = torch.zeros(num_steps + 1, num_processes, num_inputs, dtype=torch.float)
    self._rewards = torch.zeros(num_steps, num_processes, 1)
    self._value_predictions = torch.zeros(num_steps + 1, num_processes, 1)
    self._returns = torch.zeros(num_steps + 1, num_processes, 1)
    self._action_log_probabilities = torch.zeros(num_steps, num_processes, 1)

    self._actions = torch.zeros(num_steps, num_processes, 1, dtype=torch.long)
    self._possible_actionss = torch.zeros(num_steps + 1, num_processes, num_actions)

    self._not_dones = torch.ones(num_steps + 1, num_processes, 1)
    self.num_steps = num_steps
    self.step = 0
    self.device = device

  def to(self, device: torch.device):
    self._observations = self._observations.to(device)
    self._rewards = self._rewards.to(device)
    self._value_predictions = self._value_predictions.to(device)
    self._returns = self._returns.to(device)
    self._action_log_probabilities = self._action_log_probabilities.to(device)
    self._actions = self._actions.to(device)
    self._possible_actionss = self._possible_actionss.to(device)
    self._not_dones = self._not_dones.to(device)

  def store_first_transition(self, observations: torch.FloatTensor, possible_actions: List[torch.FloatTensor],
                             not_done: torch.FloatTensor = None):
    self._observations[0].copy_(observations)
    self._possible_actionss[0].copy_(possible_actions)
    if not_done is not None:
      self._not_dones[0].copy_(not_done)

    self.to(self.device)

  def roll_over_last_transition(self):
    self.store_first_transition(self._observations[-1], self._possible_actionss[-1], self._not_dones[-1])

  def insert(self, observations: torch.FloatTensor, actions: torch.LongTensor, action_log_probs: torch.FloatTensor,
             value_preds: torch.FloatTensor, rewards: torch.FloatTensor, not_dones: torch.FloatTensor,
             possible_actions: List[torch.FloatTensor]):

    self._observations[self.step + 1].copy_(observations)
    self._possible_actionss[self.step + 1].copy_(possible_actions)

    self._actions[self.step].copy_(actions)
    self._action_log_probabilities[self.step].copy_(action_log_probs)
    self._value_predictions[self.step].copy_(value_preds)
    self._rewards[self.step].copy_(rewards)
    self._not_dones[self.step + 1].copy_(not_dones)

    self.step = (self.step + 1) % self.num_steps

  def compute_returns(self, next_value: torch.FloatTensor):
    # TODO review me @d3sm0
    assert next_value.dtype == torch.float32
    assert next_value.size(1) == 1
    self._value_predictions[-1] = next_value
    gae = 0
    for step in reversed(range(self._rewards.size(0))):
      delta = self._rewards[step] + self.gamma * self._value_predictions[step + 1] * self._not_dones[step + 1] - \
              self._value_predictions[step]
      gae = delta + self.gamma * self.tau * self._not_dones[step + 1] * gae
      self._returns[step] = gae + self._value_predictions[step]

  def feed_forward_generator(self, advantages: torch.FloatTensor, num_mini_batch: int) -> (
    torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
    torch.FloatTensor, torch.FloatTensor):
    assert advantages.size() == self._rewards.size()
    assert num_mini_batch > 0
    num_steps, num_processes = self._rewards.size()[:2]
    batch_size = num_processes * num_steps
    assert batch_size >= num_mini_batch, (
      "PPO requires the number of processes ({}) "
      "* number of steps ({}) = {} "
      "to be greater than or equal to the number of PPO mini batches ({})."
      "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))

    mini_batch_size = batch_size // num_mini_batch
    sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    for indices in sampler:
      obs_batch = self._observations[:-1].view(-1, *self._observations.size()[2:])[indices]
      actions_batch = self._actions.view(-1, self._actions.size(-1))[indices]
      value_preds_batch = self._value_predictions[:-1].view(-1, 1)[indices]
      return_batch = self._returns[:-1].view(-1, 1)[indices]
      old_action_log_probs_batch = self._action_log_probabilities.view(-1, 1)[indices]
      adv_targ = advantages.view(-1, 1)[indices]

      possible_actions = self._possible_actionss[:-1].view(-1, *self._possible_actionss.size()[2:])[indices]

      yield (obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ,
             possible_actions)

  def get_observation(self, step) -> (torch.FloatTensor, torch.FloatTensor):
    assert specs.check_positive_type(step, int, strict=False)
    return self._observations[step], self._possible_actionss[step]

  def get_last_observation(self) -> torch.FloatTensor:
    return self._observations[-1]

  def get_advantages(self):
    return self._returns[:-1] - self._value_predictions[:-1]
