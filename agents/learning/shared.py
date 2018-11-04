import numpy as np
import math
import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_actions):
      if len(state.shape) == 1:
        assert len(next_state.shape) == 1
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done, next_actions))
      else:
        nr_observations = state.shape[0]
        # action = np.array(action).repeat(nr_observations, axis=0).reshape(-1, 1)
        # reward = np.array(reward).repeat(nr_observations, axis=0).reshape(-1, 1)
        # done = np.array(done).repeat(nr_observations, axis=0).reshape(-1, 1)

        # TODO: this can be more efficient by working on the indexes
        for idx in range(nr_observations):
          self.buffer.append((state[idx], action, reward, next_state[idx], done, next_actions))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_actions= zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), reward, np.array(next_state), done, next_actions

    def __len__(self):
        return len(self.buffer)


def epsilon_by_frame(
        frame_idx,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=5000,
):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


def plot(frame_idx, rewards, losses, win_ratio, action_stats, epsilon):
    # TODO: tensorboard this
    total = sum(action_stats.values())
    for k, a in action_stats.items():
      print(k, a)

    print('frame {}\n win%: {}\n losses: {}\n rewards: {}\n epsilon: {}'.format(
      frame_idx,
      win_ratio,
      np.mean(losses),
      np.mean(rewards),
      epsilon
    ))
