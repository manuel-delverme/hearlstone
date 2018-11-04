import numpy as np
import math
import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_actions):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done, next_actions))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_actions= zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.array(action).reshape(-1, 1), reward, np.concatenate(next_state), done, next_actions

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
    total = sum(action_stats)
    print({i: a/total for i, a in enumerate(action_stats)})
    print('frame {}\n win%: {}\n losses: {}\n rewards: {}\n epsilon: {}'.format(
      frame_idx,
      win_ratio,
      np.mean(losses),
      np.mean(rewards),
      epsilon
    ))
