import random

import numpy as np

from agents.heuristic.hand_coded import HeuristicAgent
from environments.vanilla_hs import VanillaHS

env = VanillaHS(skip_mulligan=True)
env.set_opponents(HeuristicAgent(level=1))


def HSenv_test():
  s0, reward, terminal, info = env.reset()
  for _ in range(3):
    done = False
    r = None
    while not done:
      possible_actions = np.argwhere(info['possible_actions'])
      random_act = random.choice(possible_actions)
      s, r, done, info = env.step(random_act)
    assert r != 0.0


def test_loss():
  env.reset()
  for _ in range(3):
    done = False
    r = None
    while not done:
      s, r, done, info = env.step(0)
      env.render()
    assert r == -1


test_loss()
