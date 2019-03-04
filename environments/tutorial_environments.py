import logging
import gym.spaces
import pprint
import collections
import functools
import numpy as np

import hs_config
import fireplace
import fireplace.logging
import hearthstone
from fireplace.exceptions import GameOver
from fireplace.game import Game, PlayState

from environments import simulator
from environments import base_env
from shared import utils

from typing import Tuple, List
from baselines.common.running_mean_std import RunningMeanStd

import environments.vanilla_hs
import agents.heuristic.hand_coded


class TradingHS(environments.vanilla_hs.VanillaHS):
  def __init__(
    self,
    minions_in_board: int = 0,
  ):
    super(TradingHS, self).__init__(
      max_cards_in_board=hs_config.VanillaHS.max_cards_in_board,
      max_cards_in_hand=0,
      skip_mulligan=True,
      starting_hp=hs_config.VanillaHS.starting_hp,
    )
    self.opponent = agents.heuristic.hand_coded.PassingAgent()
    self.minions_in_board = minions_in_board

  def reinit_game(self):
    super(TradingHS, self).reinit_game()
    # generate opponent minions
    # generate player minions

  def game_value(self):
    # if opponent minions:
    # -1
    # else:
    # nr my minions
    raise NotImplementedError
