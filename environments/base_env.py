from abc import ABC, abstractproperty, abstractmethod

import gym


class BaseEnv(gym.Env, ABC):
  class GameActions(object):
    PASS_TURN = 0

  @abstractmethod
  @property
  def action_space(self):
    raise NotImplemented

  @abstractmethod
  @property
  def cards_in_hand(self):
    raise NotImplemented

  @abstractmethod
  def play_opponent_turn(self):
    raise NotImplemented

  @abstractmethod
  def game_value(self):
    raise NotImplemented
