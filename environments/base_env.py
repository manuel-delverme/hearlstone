from abc import ABC, abstractmethod
from enum import Enum

import gym


class BaseEnv(gym.Env, ABC):
  class GameActions(Enum):
    PASS_TURN = 0

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
