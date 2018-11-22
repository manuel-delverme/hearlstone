import gym
from typing import Tuple
import gym.envs


class GymWrapper(gym.Env):
  def __init__(self, env):
    self.env = env

  @property
  def action_space(self):
    sources = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    targets = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    return gym.spaces.Discrete(
      (sources * targets) + 1
    )

  @property
  def observation_space(self):
    obs_size = self.env.observation_size
    return gym.spaces.Box(low=-1, high=1, shape=(obs_size,))

  def reset(self):
    o, r, t, i = self.env.reset()
    return o

  all_actions = {}
  def _decode_action(self, action: int) -> Tuple[int, int]:
    NUM = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    src, tar = divmod(action, NUM)
    if src == tar == NUM:
      src, tar = -1, -1
    if action not in self.all_actions:
      self.all_actions[action] = (src, tar)
      print(len(self.all_actions), max(self.all_actions.keys()))
    else:
      assert self.all_actions[action] == (src, tar)
    return src, tar

  def _encode_action(self, src: int, tar: int) -> int:
    NUM = self.env.simulation._MAX_CARDS_IN_BOARD + 1
    if src == tar == -1:
      action = NUM * NUM
    else:
      action = src * tar
    return action

  def step(self, action: int):
    src, tar = self._decode_action(action)
    transition = self.env.step((src, tar))
    return transition
