from abc import ABC, abstractproperty, abstractmethod

import gym


class BaseEnv(gym.Env, ABC):

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
