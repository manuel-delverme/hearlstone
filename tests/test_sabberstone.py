import random

import numpy as np

from agents.heuristic.hand_coded import HeuristicAgent, PassingAgent, SabberAgent
from agents.heuristic.random_agent import RandomAgent
# from environments.vanilla_hs import VanillaHS

from environments.sabber_hs import Sabbertsone
# env = VanillaHS(skip_mulligan=True)
env = Sabbertsone()
env.set_opponents([SabberAgent(level=6)])


def HSenv_test():
  s0, reward, terminal, info = env.reset()
  for _ in range(3):
    done = False
    r = None
    while not done:
      # possible_actions = np.argwhere(info['possible_actions'])
      random_act = random.choice(info['possible_actions'])
      s, r, done, info = env.step(int(random_act))
      assert env.game.CurrentPlayer.id == 1
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


# test_loss()
HSenv_test()