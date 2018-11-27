import gym
import functools
import numpy as np
from typing import Tuple
import gym.envs


class GymWrapper(gym.Env):
  def __init__(self, env):
    self.env = env
    self.all_actions = {}

  @property
  def action_space(self):
    # + hero w/weapon
    sources = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    # + hero w/weapon + play on board
    targets = self.env.simulation._MAX_CARDS_IN_BOARD + 1 + 1
    # + pass
    return gym.spaces.Discrete(
      (sources * targets) + 1
    )

  @property
  def observation_space(self):
    obs_size = self.env.observation_space
    return gym.spaces.Box(low=-1, high=1, shape=(self.action_space.n + obs_size,))

  def reset(self):
    transition = self.env.reset()
    transition = self.process_transition(transition)
    return transition[0]

  def _decode_action(self, action: int) -> Tuple[int, int]:
    # if action == self.action_space.n - 1:
    #   src, tar = -1, -1
    # else:
    #   NUM = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    #   src, tar = divmod(action, NUM)
    #   # src -= 1
    #   tar -= 1

    # if action not in self.all_actions:
    #   print('ACTION', action)
    #   for k, v in self.all_actions.items():
    #     print(k, v)
    #   raise ValueError('Decodin something that was never encoded')
    # else:
    #   assert self.all_actions[action] == (src, tar)
    src, tar = self.all_actions[action]
    return src, tar

  @functools.lru_cache(maxsize=1000)
  def _encode_action(self, src: int, tar: int) -> int:
    # # 1 is the hero
    # assert -1 <= src < self.env.simulation._MAX_CARDS_IN_BOARD + 1

    # num_srcs = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    # # num_tars = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    # if src == tar == -1:
    #   # wrap around
    #   action = self.action_space.n - 1
    # else:
    #   tar += 1
    #   assert src >= 0 and tar >= 0
    #   action = src * num_srcs + tar

    # self.all_actions[action] = (src, tar)
    action = len(self.all_actions)
    self.all_actions[action] = (src, tar)
    return action

  def step(self, action: int):
    src, tar = self._decode_action(action)
    transition = self.env.step((src, tar))
    transition = self.process_transition(transition)
    # print(transition)
    return transition

  def process_transition(self, transition):
    observation, r, t, info = transition
    # observation.extend([0]*self.action_space.n)
    action_mask = np.ones(shape=(self.action_space.n,)) * -99
    print('allowing:', end='')
    for poss_action in info['possible_actions']:
      action_id = self._encode_action(*poss_action)
      print(action_id, end=',')
      action_mask[action_id] = 0
    print()
    observation = np.concatenate((action_mask, observation), axis=0)
    return observation, r, t, info
