import random
from collections import deque

import numpy as np


class PrioritizedBuffer(object):
  def __init__(self, capacity, prob_alpha=0.6):
    self.prob_alpha = prob_alpha
    self.capacity = capacity
    self.filled_up_to = 0
    self.buffer = np.empty(shape=(capacity, 6), dtype=np.object)
    self.pos = 0
    self.priorities = np.zeros((capacity,), dtype=np.float32)
    self.priorities[0] = 1

  def push(self, state, action, reward, next_state, done, next_actions):
    assert len(next_state.shape) == 1
    assert state.ndim == next_state.ndim

    self.buffer[self.pos, :] = np.array((state, action, reward, next_state, done, next_actions))

    # TODO: cache this
    self.priorities[self.pos] = self.priorities.max()

    self.pos = (self.pos + 1) % self.capacity
    if self.filled_up_to < self.pos:
      self.filled_up_to = self.pos

  def sample(self, batch_size, beta):
    priorities = self.priorities[:self.filled_up_to] ** self.prob_alpha
    probabilities = priorities / priorities.sum()

    indices = np.random.choice(self.filled_up_to, batch_size, p=probabilities)
    # samples = []
    states = self.buffer[indices, [0]]
    actions = self.buffer[indices, [1]]
    rewards = self.buffer[indices, [2]]
    next_states = self.buffer[indices, [3]]
    dones = self.buffer[indices, [4]]
    next_actions = self.buffer[indices, [5]]
    assert self.buffer.shape[1] == 6
    # states = np.concatenate(batch[0])
    # actions = batch[1]
    # rewards = batch[2]
    # next_states = np.concatenate(batch[3])
    # dones = batch[4]

    total = len(self.buffer)
    weights = (total * probabilities[indices]) ** (-beta)
    weights /= weights.max()
    weights = np.array(weights, dtype=np.float32)

    return states, actions, rewards, next_states, dones, next_actions, indices, weights

  def update_priorities(self, batch_indices, batch_priorities):
    for idx, prio in zip(batch_indices, batch_priorities):
      self.priorities[idx] = prio

  def __len__(self):
    return len(self.buffer)


class ReplayBuffer(object):
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done, next_actions):
    assert len(next_state.shape) == 1
    self.buffer.append((state, action, reward, next_state, done, next_actions))

  def sample(self, batch_size):
    state, action, reward, next_state, done, next_actions = zip(*random.sample(self.buffer, batch_size))
    return np.array(state), np.array(action), reward, np.array(next_state), done, next_actions

  def __len__(self):
    return len(self.buffer)
