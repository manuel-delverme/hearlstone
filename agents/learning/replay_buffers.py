import random
from collections import deque
import baselines.deepq.replay_buffer

import numpy as np


class PrioritizedBufferOpenAI(baselines.deepq.replay_buffer.PrioritizedReplayBuffer):
  def __init__(self, capacity, state_size, action_size, prob_alpha=0.6, ):
    super(PrioritizedBufferOpenAI, self).__init__(capacity, prob_alpha)

  def push(self, state, action, reward, next_state, done, next_actions):
    assert len(next_state.shape) == 1
    assert state.ndim == next_state.ndim
    super(PrioritizedBufferOpenAI, self).add(state, action, reward, next_state, done, next_actions)

  def sample(self, batch_size, beta):
    states, actions, rewards, next_states, dones, next_actions, weights, indices = super(PrioritizedBufferOpenAI, self).sample(batch_size, beta)
    return states, actions, rewards, next_states, dones, next_actions, indices, weights


class PrioritizedBuffer(object):
  def __init__(self, capacity, state_size, action_size, prob_alpha=0.6, ):
    # TODO: sumtree
    self.prob_alpha = prob_alpha
    self.capacity = capacity
    self.filled_up_to = 0
    self.pos = 0
    self.priorities = np.zeros((capacity,), dtype=np.float32)
    self.priorities[0] = 1
    sizes = [state_size, action_size, 1, state_size, 1]
    self.boundaries = [0, ]
    for s in sizes:
      self.boundaries.append(self.boundaries[-1] + s)

    self.buffer = np.empty(shape=(capacity, self.boundaries[-1]), dtype=np.float32)
    self.next_actions_buffer = [None, ] * capacity

  def _compact(self, *args):
    row = np.empty(shape=(1, self.boundaries[-1]))
    old_bound = self.boundaries[0]
    for vec, bound in zip(args, self.boundaries[1:]):
      row[0, old_bound: bound] = vec.reshape(1, -1)
      old_bound = bound
    return row

  def _decompact(self, vec):
    old_bound = self.boundaries[0]
    retr = []
    for bound in self.boundaries[1:]:
      retr.append(vec[:, old_bound: bound])
      old_bound = bound
    return retr

  def push(self, state, action, reward, next_state, done, next_actions):
    assert len(next_state.shape) == 1
    assert state.ndim == next_state.ndim

    self.buffer[self.pos, :] = self._compact(state, action, reward, next_state, done)
    self.next_actions_buffer[self.pos] = next_actions

    # TODO: cache this
    self.priorities[self.pos] = self.priorities.max()

    self.pos = (self.pos + 1) % self.capacity
    if self.filled_up_to < self.pos:
      self.filled_up_to = self.pos

  def sample(self, batch_size, beta):
    priorities = self.priorities[:self.filled_up_to] ** self.prob_alpha
    probabilities = priorities / priorities.sum()

    indices = np.random.choice(self.filled_up_to, batch_size, p=probabilities)
    states, actions, rewards, next_states, dones = self._decompact(self.buffer[indices])
    next_actions = [self.next_actions_buffer[idx] for idx in indices]

    total = self.__len__()
    weights = (total * probabilities[indices]) ** (-beta)
    weights /= weights.max()
    weights = np.array(weights, dtype=np.float32).reshape(-1, 1)

    return states, actions, rewards, next_states, dones, next_actions, indices, weights

  def update_priorities(self, batch_indices, batch_priorities):
    for idx, prio in zip(batch_indices, batch_priorities):
      self.priorities[idx] = prio

  def __len__(self):
    return self.filled_up_to


class ReplayBuffer(object):
  def __init__(self, capacity, state_size, action_size, prob_alpha=None, ):
    raise DeprecationWarning('replace w/ prioritized and uniform probs')
    self.buffer = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done, next_actions):
    assert len(next_state.shape) == 1
    self.buffer.append((state, action, reward, next_state, done, next_actions))

  def sample(self, batch_size, beta):
    state, action, reward, next_state, done, next_actions = zip(*random.sample(self.buffer, batch_size))
    return np.array(state), np.array(action), reward, np.array(next_state), done, next_actions

  def __len__(self):
    return len(self.buffer)
