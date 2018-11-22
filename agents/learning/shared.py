import numpy as np
import math


def epsilon_schedule(
        epsilon_start=1.0,
        epsilon_final=0.1,
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


def sync_target(source_network, target_network):
  target_network.load_state_dict(source_network.state_dict())
