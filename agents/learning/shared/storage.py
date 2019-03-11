from typing import Sequence
import gym.spaces

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
  def __init__(self, num_steps: int, num_processes: int, obs_shape: Sequence[int], action_space: gym.spaces.Discrete):
    assert num_steps > 0
    assert num_processes > 0
    assert len(obs_shape) == 1
    assert isinstance(action_space, gym.spaces.Discrete)

    self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
    self.rewards = torch.zeros(num_steps, num_processes, 1)
    self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
    self.returns = torch.zeros(num_steps + 1, num_processes, 1)
    self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

    self.actions = torch.zeros(num_steps, num_processes, 1, dtype=torch.long)
    self.possible_actionss = torch.zeros(num_steps + 1, num_processes, action_space.n)

    self.masks = torch.ones(num_steps + 1, num_processes, 1)
    self.num_steps = num_steps
    self.step = 0

  def to(self, device: torch.device):
    self.obs = self.obs.to(device)
    self.rewards = self.rewards.to(device)
    self.value_preds = self.value_preds.to(device)
    self.returns = self.returns.to(device)
    self.action_log_probs = self.action_log_probs.to(device)
    self.actions = self.actions.to(device)
    self.possible_actionss = self.possible_actionss.to(device)
    self.masks = self.masks.to(device)

  def insert(self, observations: torch.FloatTensor, actions: torch.LongTensor, action_log_probs: torch.FloatTensor,
    value_preds: torch.FloatTensor, rewards: torch.FloatTensor, masks: torch.FloatTensor,
    possible_actions: torch.FloatTensor):

    self.obs[self.step + 1].copy_(observations)
    self.possible_actionss[self.step + 1].copy_(possible_actions)

    self.actions[self.step].copy_(actions)
    self.action_log_probs[self.step].copy_(action_log_probs)
    self.value_preds[self.step].copy_(value_preds)
    self.rewards[self.step].copy_(rewards)
    self.masks[self.step + 1].copy_(masks)

    self.step = (self.step + 1) % self.num_steps

  def update_for_new_rollouts(self):
    self.obs[0].copy_(self.obs[-1])
    self.masks[0].copy_(self.masks[-1])
    self.possible_actionss[0].copy_(self.possible_actionss[-1])

  def compute_returns(self, next_value: torch.FloatTensor, use_gae: bool, gamma: float, tau: float):
    assert isinstance(next_value, torch.FloatTensor)
    assert next_value.size(1) == 1
    assert isinstance(use_gae, bool)
    assert isinstance(gamma, float)
    assert isinstance(tau, float)
    if use_gae:
      self.value_preds[-1] = next_value
      gae = 0
      for step in reversed(range(self.rewards.size(0))):
        delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
        gae = delta + gamma * tau * self.masks[step + 1] * gae
        self.returns[step] = gae + self.value_preds[step]
    else:
      self.returns[-1] = next_value
      for step in reversed(range(self.rewards.size(0))):
        self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

  def feed_forward_generator(self, advantages: torch.FloatTensor, num_mini_batch: int) -> (
    torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor,
    torch.FloatTensor, torch.FloatTensor):
    assert advantages.size() == self.rewards.size()
    assert num_mini_batch > 0
    num_steps, num_processes = self.rewards.size()[:2]
    batch_size = num_processes * num_steps
    assert batch_size >= num_mini_batch, (
      "PPO requires the number of processes ({}) "
      "* number of steps ({}) = {} "
      "to be greater than or equal to the number of PPO mini batches ({})."
      "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))

    mini_batch_size = batch_size // num_mini_batch
    sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
    for indices in sampler:
      obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
      actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
      value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
      return_batch = self.returns[:-1].view(-1, 1)[indices]
      old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
      adv_targ = advantages.view(-1, 1)[indices]

      possible_actions = self.possible_actionss[:-1].view(-1, *self.possible_actionss.size()[2:])[indices]

      yield (obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ,
             possible_actions)
