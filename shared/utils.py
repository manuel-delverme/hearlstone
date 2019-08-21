import collections
import glob
import logging
import os
import pickle
import random
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from functools import _lru_cache_wrapper
from functools import lru_cache
from typing import Callable, List, Tuple, Union

import numpy as np
import tqdm
from hearthstone.enums import CardClass, CardType
from numpy import ndarray

import hs_config
import pysabberstone.python.rpc.python_pb2 as sabberstone_protobuf
import shared.constants as C
from agents.base_agent import Agent
from environments import base_env


class HSLogger(logging.Logger):
  formatter = logging.Formatter('%(asctime)s -  %(processName)s - %(name)s - %(levelname)s - %(message)s')
  save_path = f'/tmp/heaRLstone-{hs_config.comment}.log'

  def __init__(self, name: str, log_to_stdout: bool = True):
    assert isinstance(name, str)
    super(HSLogger, self).__init__(logging.getLogger(name))
    self.timers = collections.defaultdict(lambda: [0, 0])
    self.current_timer = None

    handler = logging.FileHandler(self.save_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(self.formatter)
    self.addHandler(handler)

    if log_to_stdout:
      handler = logging.StreamHandler()
      handler.setLevel(logging.INFO)
      handler.setFormatter(self.formatter)
      self.addHandler(handler)
    self.info(f"Timer for {name} saves  in {self.save_path}")

    self.stat_file = tempfile.mktemp(prefix='heaRLogs/')
    self.info(f"Stats for {name} saves  in {self.stat_file}")
    self.saved_stats = False

  def __call__(self, timer_name: str) -> object:
    timer_name = "#" + timer_name.upper() + "#"
    self.current_timer = timer_name
    return self

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, type, value, traceback):
    delta = time.time() - self.start
    self.timers[self.current_timer][0] += delta
    self.timers[self.current_timer][1] = delta
    self.info(self.__str__())

  def __str__(self):
    cum, last = self.timers[self.current_timer]
    return f"{self.current_timer} - (total_time, delta) ({cum:.4f},{last:.4f})"

  def __del__(self):
    for h in self.handlers:
      if isinstance(h, logging.FileHandler):
        h.close()

  if not hs_config.Environment.ENV_DEBUG_METRICS:
    def __call__(self, timer_name: str) -> object:
      return self

    def __enter__(self):
      pass

    def __exit__(self, type, value, traceback):
      pass

    def log_stats(self, stats: dict):
      pass

    def info(self, *args, **kwargs):
      pass


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
  player_order = [player_policy, opponent_policy]
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

    game_value = environment.agent_game_vale()

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


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
  """Decreases the learning rate linearly"""
  lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
  weight_init(module.weight.data, gain=gain)
  bias_init(module.bias.data)
  return module


def can_autoreset(auto_reset, game_ref):
  # the opponent cannot auto_reset
  if not auto_reset:
    return True
  if game_ref.CurrentPlayer.id == C.AGENT_ID:
    return True
  if game_ref.state == sabberstone_protobuf.Game.COMPLETE:
    return True
  return False


def load_latest_checkpoint(checkpoint=None, experiment_id=hs_config.comment):
  if checkpoint is None:
    print('loading checkpoints', hs_config.PPOAgent.save_dir + f'/*{experiment_id}*')
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + f'/*{experiment_id}*')
    if checkpoints:
      # specs = list(map(get_ckpt_specs, checkpoints))
      # checkpoint_files = sorted(zip(checkpoints, specs), key=lambda x: (int(x[1].score), int(x[1].steps)))
      # checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=:)", x).group(0)))
      checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=\.pt)", x).group(0)))
      checkpoint = checkpoint_files[-1]
  print('found', checkpoint)
  return checkpoint