import itertools
import random
import time

import numpy as np
from tqdm import tqdm

import agents.heuristic.random_agent
# from environments.vanilla_hs import VanillaHS
from environments.sabber_hs import Sabbertsone

# env = VanillaHS(skip_mulligan=True)
env = Sabbertsone('localhost:50052')
# env.set_opponents([SabberAgent(level=6)])


def HSenv_test():
  s0, reward, terminal, info = env.reset()
  avg_time = []
  for _ in range(int(1e4)):
    done = False
    r = None
    while not done:
      random_act = random.choice(info['possible_actions'])
      start = time.time()
      s, r, done, info = env.step(int(random_act))
      delta = time.time() - start
      avg_time.append(1 / delta)
      # assert env.game.CurrentPlayer.id == 1
    # assert r != 0.0
    print(np.mean(avg_time))


def test_wrapperFPS():
  env.set_opponents([agents.heuristic.random_agent.RandomAgent()])
  s0, reward, terminal, info = env.reset()
  for _ in tqdm(itertools.count()):
    pa = info['possible_actions']
    rows = np.argwhere(pa)  # row is always 0
    random_act = random.choice(rows)
    s, r, done, info = env.step(int(random_act))
    if done:
      env.reset()


def test_loss():
  env.reset()
  for _ in range(3):
    done = False
    r = None
    while not done:
      s, r, done, info = env.step(0)
      env.render()
    assert r == -1


test_wrapperFPS()
# test_loss()
# HSenv_test()
