from abc import ABC, abstractproperty, abstractmethod

import gym


class BaseEnv(gym.Env, ABC):
  class GameActions(object):
    PASS_TURN = 0

  @property
  @abstractmethod
  def action_space(self):
    raise NotImplemented

  @property
  @abstractmethod
  def observation_space(self):
    raise NotImplemented

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
