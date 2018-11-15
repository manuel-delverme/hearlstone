import numpy as np
import math
import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_actions):
      assert len(next_state.shape) == 1
      self.buffer.append((state, action, reward, next_state, done, next_actions))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_actions= zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), reward, np.array(next_state), done, next_actions

    def __len__(self):
        return len(self.buffer)


def epsilon_schedule(
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=200,
):
  step_nr = 0
  while True:
    step_nr += 1
    yield epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step_nr / epsilon_decay)


def plot(frame_idx, rewards, losses, win_ratio, action_stats, epsilon, end_turns):
    # TODO: tensorboard this
    total = sum(action_stats.values())
    for k, a in action_stats.items():
      print(k, a)

    print('frame {}\n win%: {}\n end turn: {}\n losses: {}\n rewards: {}\n epsilon: {}'.format(
      frame_idx,
      win_ratio,
      np.mean(end_turns),
      np.mean(losses),
      np.mean(rewards),
      epsilon
    ))
