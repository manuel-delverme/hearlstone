#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environments.sabber_hs import Sabberstone
from agents.heuristic.hand_coded import SabberAgent
import shared.constants as C
import random
import numpy as np
import torch

def test_gui():
  import torch.distributions as dist
  env = Sabberstone('0.0.0.0:50052')
  env.set_opponents([SabberAgent()])
  s0, reward, terminal, info = env.reset()
  pi = dist.Categorical(probs= torch.tensor((1/C._ACTION_SPACE, ) * C._ACTION_SPACE))
  while not terminal:
    pa = np.argwhere(info['possible_actions'])  # row is always 0
    action = random.choice(pa)
    s, r, done, info = env.step(int(action))
    value = np.random.randint(-1000, 1000)
    env.render(choice=action, action_distribution=pi, value=value)
    if done:
      env.render(choice=action, action_distribution=pi, value=value, reward=r)
      s0, _, _, info = env.reset()


if __name__ == "__main__":
  test_gui()



