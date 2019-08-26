#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hs_config
from agents.heuristic.random_agent import RandomAgent
from game_utils import GameManager, Ladder

elo = Ladder()
manager = GameManager(hs_config.Environment.address)
manager.use_heuristic_opponent = False
manager.opponents = ["random"] * 3
env = manager(env_number=1)
agent = RandomAgent()
from collections import defaultdict
for _ in range(10):
  scores = defaultdict(lambda: [])
  for ep in range(10):
    o, _, _, info = env.reset()

    while True:
      a = agent.choose(o, info)
      o, r, d, info, = env.step(a)
      if d:
        scores[env.current_k].append(r)
        break
  manager.update_score(scores)
  manager.add_learned_opponent('default')
  env = manager()


print(elo.games_count)


