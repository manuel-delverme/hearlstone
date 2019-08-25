#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hs_config
from agents.heuristic.random_agent import RandomAgent
from game_utils import GameManager, Ladder

elo = Ladder()
manager = GameManager(hs_config.Environment.address)
manager.use_heuristic_opponent = True
env = manager()
agent = RandomAgent()

scores ={0:[]} # idx, list of rewards
for ep in range(10):
  o, _, _, info = env.reset()

  while True:
    a = agent.choose(o, info)
    o, r, d, info, = env.step(a)
    if d:
      scores[0].append(r)
      break

elo.update(scores)


