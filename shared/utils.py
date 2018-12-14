import pickle
from contextlib import contextmanager
import sys
import os

import numpy as np
import tqdm
from environments import base_env
import sys
import os
from functools import lru_cache
import hashlib
from typing import List

from agents import base_agent
import random

from agents.base_agent import Agent
from hearthstone.enums import CardClass, CardType

from functools import _lru_cache_wrapper
from numpy import ndarray
from typing import Callable, List, Tuple, Union
def to_tuples(list_of_lists):
  tuple_of_tuples = []
  for item in list_of_lists:
    if isinstance(item, list):
      item = to_tuples(item)
    tuple_of_tuples.append(item)
  return tuple(tuple_of_tuples)


def disk_cache(f: Callable) -> _lru_cache_wrapper:
  @lru_cache(maxsize=1024)
  def wrapper(*args, **kwargs):
    fid = f.__name__
    cache_file = "cache/{}".format(fid)
    if args:
      if not os.path.exists(cache_file):
        os.makedirs(cache_file)
      fid = fid + "/" + "::".join(str(arg) for arg in args).replace("/", "_")
      cache_file = "cache/{}".format(fid)
    cache_file += ".pkl"
    try:
      with open(cache_file, "rb") as fin:
        retr = pickle.load(fin)
    except FileNotFoundError:
      retr = f(*args, **kwargs)
      with open(cache_file, "wb") as fout:
        pickle.dump(retr, fout)
    return retr

  return wrapper


def arena_fight(
  environment: base_env.BaseEnv,
  player_policy: Agent,
  opponent_policy: Agent,
  nr_games: int = 100,
):
  player_order: List[Agent] = [player_policy, opponent_policy]
  random.shuffle(player_order)
  active_player, passive_player = player_order  # type: (Agent, Agent)

  scoreboard = {
    'won': 0,
    'lost': 0,
    'draw': 0
  }
  for nr_game in tqdm.tqdm(range(nr_games)):
    state, reward, terminal, info = environment.reset()
    assert reward == 0.0
    assert not terminal

    while not terminal:
      action = active_player.choose(state, info)
      environment.render()
      state, reward, terminal, info = environment.step(action)

      if action == environment.GameActions.PASS_TURN:
        active_player, passive_player = passive_player, active_player

    game_value = environment.game_value()

    if game_value == 1:
      scoreboard['won'] += 1
    elif game_value == -1:
      scoreboard['lost'] += 1
    elif game_value == 0:
      scoreboard['draw'] += 1
      raise ValueError
    else:
      raise ValueError

  win_ratio = float(scoreboard['won']) / sum(scoreboard.values())
  return win_ratio


def random_draft(card_class: CardClass, exclude: Tuple = tuple(), deck_length: int = 30, max_mana: int = 30) -> List[
  str]:
  from fireplace import cards

  deck = []
  collection = []
  for card_id, card_obj in cards.db.items():
    if card_obj.description != '':
      continue
    if card_id in exclude:
      continue
    if not card_obj.collectible:
      continue
    # Heroes are collectible...
    if card_obj.type == CardType.HERO:
      continue
    if card_obj.card_class and card_obj.card_class not in (
      card_class, CardClass.NEUTRAL):
      continue
    if card_obj.cost > max_mana:
      continue
    collection.append(card_obj)

  while len(deck) < deck_length:
    card = random.choice(collection)
    if deck.count(card.id) < card.max_count_in_deck:
      deck.append(card.id)
  return deck

@contextmanager
def suppress_stdout():
  with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
      yield
    finally:
      sys.stdout = old_stdout


def one_hot_actions(actions: Union[Tuple[Tuple[int, int]], Tuple[Tuple[int, int, int]]], num_actions: int) -> ndarray:
  possible_actions = np.zeros((len(actions), num_actions), np.float32)
  for row, pas in enumerate(actions):
    for pa in pas:
      possible_actions[row, pa] = 1
  return possible_actions
