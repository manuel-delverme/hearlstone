from abc import ABC, abstractmethod
from enum import IntEnum

import gym


class BaseEnv(gym.Env, ABC):
  class GameActions(IntEnum):
    PASS_TURN = 0

  class GameOver(Exception):
    pass

  def __init__(self):
    self.opponent = None
    self.opponents = [None, ]

    self.opponent_obs_rms = None
    self.opponent_obs_rmss = [None, ]

  @property
  @abstractmethod
  def cards_in_hand(self):
    raise NotImplemented

  @abstractmethod
  def play_opponent_action(self):
    raise NotImplemented

  @abstractmethod
  def game_value(self):
    raise NotImplemented

  def set_opponents(self, opponents, opponent_obs_rmss=None):
    self.opponents = opponents
    self.opponent_obs_rmss = opponent_obs_rmss
