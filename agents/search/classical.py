from typing import Callable
import shelve
import numpy as np
import gym
import random
import tqdm

import agents.base_agent
import agents.learning.replay_buffers
import copy


# TODO use the shmem version

class DepthFirstSearchAgent(object):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.search_depth = 0
    # self.policy = {}
    self.policy = shelve.open('policy.shelf')

  def train(self, make_env: Callable[[], gym.Env], game_steps=None, checkpoint_every=10000):
    env = make_env()
    observation, reward, terminal, possible_actions = env.reset()

    rs = {0: 0, 1: 0, -1: 0}
    sequence = []
    for step_nr in tqdm.tqdm(range(100000)):
      pa, = np.where(possible_actions)
      action = self.choose(env, observation, pa)
      sequence.append((env.render(), action))
      observation, reward, done, possible_actions = env.step(action)
      rs[int(reward)] += 1

      if done:
        assert reward in (-1.0, 0.0, 1.0)
        observation, reward, terminal, possible_actions = env.reset()
        sequence.clear()
      else:
        assert reward < 1
      print(rs)

  def choose(self, original_env, obs, possible_actions):
    self.search_depth = 0
    try:
      a, r = self._choose(original_env, obs, possible_actions, 0)
    except RuntimeError:
      a = None

    print('went up to dept', self.search_depth)
    if a is None:
      a = random.choice(possible_actions)
    return a

  def _choose(self, original_env, obs, possible_actions, depth=0):
    if depth == 6:
      raise RuntimeError

    self.search_depth = max(depth, self.search_depth)
    # print('depth', self.search_depth, depth)
    obs = str(obs)
    if obs in self.policy:
      return self.policy[obs], 1.0

    next_level = []
    for a in reversed(possible_actions):
      env = copy.deepcopy(original_env)
      obs, r, t, pa = env.step(a)
      obs = str(obs)
      pa, = np.where(pa)
      if r == 1.0:
        self.policy[obs] = a
        return a, r

      if r == 0.0:
        next_level.append((a, env, obs, pa))

    for oa, env, obs, pa in next_level:
      try:
        a, r = self._choose(env, obs, pa, depth=depth + 1)
      except RuntimeError:
        pass
      else:
        if r == 1.0:
          self.policy[obs] = a
          return oa, r
    return None, -1.0
