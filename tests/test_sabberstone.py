import itertools
import random

import numpy as np
from tqdm import tqdm

from agents.heuristic.hand_coded import SabberAgent
from environments.sabber_hs import bind_address

# from environments.vanilla_hs import VanillaHS
# env = VanillaHS(skip_mulligan=True)
env = bind_address('localhost:50052')()
env.set_opponents([SabberAgent(level=6)])

import time

def HSenv_test():
  s0, reward, terminal, info = env.reset()
  avg_time = [ ]
  for _ in range(int(1e4)):
    done = False
    r = None
    while not done:

      random_act = random.choice(info['possible_actions'])
      start = time.time()
      s, r, done, info = env.step(int(random_act))
      delta = time.time() - start
      avg_time.append(1/delta)
      #assert env.game.CurrentPlayer.id == 1
    #assert r != 0.0
    print(np.mean(avg_time))


def test_wrapperFPS():
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
