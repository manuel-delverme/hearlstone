import pickle
import tqdm
from environments import base_env
import sys
import os
from functools import lru_cache
import hashlib
from typing import List
from environments import simple_env

from agents import base_agent
import random

from agents.base_agent import Agent


def to_tuples(list_of_lists):
  tuple_of_tuples = []
  for item in list_of_lists:
    if isinstance(item, list):
      item = to_tuples(item)
    tuple_of_tuples.append(item)
  return tuple(tuple_of_tuples)


def disk_cache(f):
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
  player_order = [player_policy, opponent_policy]  # type: List[Agent]
  random.shuffle(player_order)
  active_player, passive_player = player_order  # type: (Agent, Agent)

  scoreboard = {
    'won': 0,
    'lost': 0,
    'draw': 0
  }
  for nr_game in tqdm.tqdm(range(nr_games)):
    terminal = False
    state, info = environment.reset()
    possible_actions = info['possible_actions']
    while not terminal:
      action = active_player.choose(state, possible_actions)
      state, reward, terminal, info = environment.step(action)
      possible_actions = info['possible_actions']

      if (action == environment.GameActions.PASS_TURN
          or (hasattr(action, 'card') and action.card is None)
          or len(possible_actions) == 0):
        active_player, passive_player = passive_player, active_player
    game_value = environment.game_value
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

