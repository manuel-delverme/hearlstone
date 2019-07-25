import random

import numpy as np

from agents.heuristic.hand_coded import HeuristicAgent, PassingAgent
from agents.heuristic.random_agent import RandomAgent
# from environments.vanilla_hs import VanillaHS

from sb_env.SabberStone_python_client.simulator import Sabbertsone, stub
# env = VanillaHS(skip_mulligan=True)
env = Sabbertsone()
env.set_opponents(RandomAgent())


def HSenv_test():
  s0, reward, terminal, info = env.reset()
  for _ in range(3):
    done = False
    r = None
    while not done:
      # possible_actions = np.argwhere(info['possible_actions'])
      random_act = random.choice(info['possible_actions'])
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
HSenv_test()