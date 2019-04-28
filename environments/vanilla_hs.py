import collections
import logging
import pprint
import random
from typing import Tuple, Dict, Text, Any, Iterable

import baselines.common.running_mean_std
import fireplace
import fireplace.logging
import gym.spaces
import hearthstone
import numpy as np
import torch
from fireplace.exceptions import GameOver
from fireplace.game import PlayState

import agents.base_agent
import hs_config
from environments import base_env
from environments import simulator
from shared import utils


def get_hp(card):
  try:
    return card.hp
  except AttributeError:
    return card.health


def episodic_log(func):
  if not hs_config.DEBUG:
    return func

  def wrapper(*args, **kwargs):
    self = args[0]

    func_name = func.__name__
    new_episode = func_name == 'reinit_game'

    if new_episode:
      self.log.append([])
      self.log_call_depth = 0
    extra_info = tuple()  # (inspect.stack()[1:], r)

    pre = tuple(''.join((' ',) * self.log_call_depth))
    self.log[-1].append(tuple(pre + (func_name,) + args[1:] + tuple(kwargs) + extra_info))

    self.log_call_depth += 1
    retr = func(*args, **kwargs)
    self.log_call_depth -= 1

    pre = tuple(''.join((' ',) * self.log_call_depth))
    self.log[-1].append(tuple(pre + (func_name,) + (retr,) + extra_info))
    return retr

  return wrapper


class VanillaHS(base_env.BaseEnv):
  def __init__(
    self,
    max_cards_in_board=simulator.HSsimulation._MAX_CARDS_IN_BOARD,
    max_cards_in_hand=simulator.HSsimulation._MAX_CARDS_IN_HAND,
    skip_mulligan=True,
    cheating_opponent=False,
    starting_hp=hs_config.VanillaHS.starting_hp,
    sort_decks=hs_config.VanillaHS.sort_decks,
    seed=None,
    extra_seed=None,
  ):
    """
    A game with only vanilla monster cars, mage+warrior hero powers, 2 cards
    in front of each player.

    Args:
      starting_hp:
    """
    if seed is not None:
      random.seed(seed + extra_seed)
      np.random.seed(seed + extra_seed)
    self.opponent = None
    self.opponents = [None, ]
    self.opponent_obs_rms = None
    self.opponent_obs_rmss = [None, ]

    self.gui = None
    self.last_info = None
    self.id_to_action = [(-1, -1), ]
    self.action_history = collections.deque(maxlen=2)
    self.action_history.append([])

    for src_idx in range(max_cards_in_hand):
      self.id_to_action.append((src_idx, -1))
    for src_idx in range(max_cards_in_board):
      for target_idx in range(max_cards_in_board + 1):
        self.id_to_action.append((src_idx, target_idx))

    self.action_to_id_dict = {v: k for k, v in enumerate(self.id_to_action)}
    self.num_actions = len(self.id_to_action)

    fireplace.cards.db.initialized = True
    print("Initializing card database")
    db, xml = hearthstone.cardxml.load()
    self.log_call_depth = 0
    if hs_config.DEBUG:
      self.log = collections.deque(maxlen=2)
      logger = logging.getLogger('fireplace')
      logger.setLevel(logging.INFO)
    else:
      self.log = None
      logging.getLogger("fireplace").setLevel(logging.ERROR)

    # allow coin, mage and warrior hero powers and moonfire (to cheat)
    allowed_ids = ("GAME_005", "CS2_034", "CS2_102", "CS2_008")
    for id, card in db.items():
      if card.description == "" or card.id in allowed_ids:
        fireplace.cards.db[id] = fireplace.cards.db.merge(id, card)

    simulator.HSsimulation._MAX_CARDS_IN_BOARD = max_cards_in_board
    simulator.HSsimulation._MAX_CARDS_IN_HAND = max_cards_in_hand
    self.max_cards_in_board = max_cards_in_board

    self.skip_mulligan = skip_mulligan
    self.lookup_action_id_to_obj = {}
    self.cheating_opponent = cheating_opponent
    self.sort_decks = sort_decks
    self.starting_hp = starting_hp
    self.minions_in_board = 0

    self.reinit_game()

    obs, _, _, _ = self.gather_transition(autoreset=False)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=obs.shape, dtype=np.int)
    self.action_space = gym.spaces.Discrete(self.num_actions)

  def dump_log(self, log_file):
    if self.log is None:
      return

    with open(log_file, 'w') as fout:
      for episode in self.log:
        fout.write('=' * 100)
        fout.write('\n')
        for event in episode:
          fout.write(pprint.pformat(event))
          fout.write('\n')

  @episodic_log
  def encode_actions(self, actions: Tuple[simulator.HSsimulation.Action]):
    assert isinstance(actions, tuple)
    assert isinstance(actions[0], simulator.HSsimulation.Action)

    self.lookup_action_id_to_obj.clear()
    encoded_actions = []
    for action in actions:
      action_id = self.action_to_id(action)
      self.lookup_action_id_to_obj[action_id] = action
      encoded_actions.append(action_id)
    return tuple(encoded_actions)

  @episodic_log
  def decode_action(self, encoded_action: int) -> simulator.HSsimulation.Action:
    assert isinstance(encoded_action, (int, np.int64))
    return self.lookup_action_id_to_obj[encoded_action]

  @episodic_log
  def action_to_id(self, possible_action) -> Tuple:

    if possible_action.card is None:
      source_idx = -1
      target_idx = -1
    else:
      card = possible_action.card
      target = possible_action.params['target']
      if target is None:
        # playing from hand
        source_idx = card.controller.hand.index(card)
        target_idx = -1
      else:
        # trading
        source_idx = card.controller.field.index(card)
        # 0 for hero, 1+ board
        target_idx = target.zone_position
    try:
      action_id = self.action_to_id_dict[(source_idx, target_idx)]
      return action_id
    except KeyError as e:
      print(possible_action)
      raise e

  @episodic_log
  def reinit_game(self):
    self.simulation = simulator.HSsimulation(
      skip_mulligan=self.skip_mulligan,
      cheating_opponent=self.cheating_opponent,
      starting_hp=self.starting_hp,
      sort_decks=self.sort_decks,
    )
    if random.random() < hs_config.VanillaHS.old_opponent_prob:
      opponent_idx = random.randint(0, len(self.opponents) - 1)
    else:
      opponent_idx = -1

    self.opponent = self.opponents[opponent_idx]
    self.opponent_obs_rms = self.opponent_obs_rmss[opponent_idx]

    try:
      self.opponent.set_simulation(self.simulation)
    except AttributeError:
      pass
    self.games_played = 0
    self.episode_steps = 0
    self.games_finished = 0
    self.info = None

  @episodic_log
  def game_value(self):
    player = self.simulation.player
    if player.playstate == PlayState.WON:
      return 1
    elif player.playstate == PlayState.LOST:
      return 0
    elif player.playstate == PlayState.TIED:
      raise ValueError
    elif player.playstate == PlayState.INVALID:
      raise ValueError
    elif player.playstate == PlayState.PLAYING:
      raise ValueError
    elif player.playstate == PlayState.WINNING:
      raise ValueError
    elif player.playstate == PlayState.LOSING:
      raise ValueError
    elif player.playstate == PlayState.DISCONNECTED:
      raise ValueError
    elif player.playstate == PlayState.CONCEDED:
      raise ValueError
    else:
      raise ValueError("{} not a valid playstate to eval game_value".format(
        self.simulation.player.playstate))

  @property
  @episodic_log
  def cards_in_hand(self):
    return self.simulation._MAX_CARDS_IN_HAND

  def render(self, mode='human'):
    if self.gui is None:
      import gui
      self.gui = gui.GUI()

    obs = self.last_info['observation']
    offset, board, hand, mana = self.render_player(obs)
    self.gui.draw_agent(board=board, hand=hand, mana=mana)

    offset, board, hand, mana = self.render_player(obs, offset)
    self.gui.draw_opponent(board=board, hand=hand, mana=mana)

    info = self.last_info.copy()
    info['possible_actions'] = np.argwhere(info['possible_actions']).flatten()
    action_history = info['action_history']
    del info['action_history']
    del info['observation']
    self.gui.log(action_history[0], row=1)
    if len(action_history) > 1:
      self.gui.log(action_history[1], row=2)
    self.gui.log(" ".join(("{}:{}".format(k, v) for k, v in info.items())), row=3, multiline=True)

    if mode == 'human':
      self.gui.screen.getch()
    else:
      raise NotImplementedError

  def render_player(self, obs, offset=0):
    hand = []
    for minion_number in range(self.simulation._MAX_CARDS_IN_HAND):
      card = obs[offset: offset + 3]
      if card[:2].max() > -1:
        hand.append(list(int(c) for c in card))
      offset += 3

    board = []
    for minion_number in range(self.max_cards_in_board + 1):
      card = obs[offset: offset + 3]
      if card.max() > -1:
        board.append(list(int(c) for c in card))
      offset += 3

    mana = obs[offset]
    offset += 1
    return offset, [board[-1], ] + board[:-1], hand, mana

  @episodic_log
  def reset(self):
    self.reinit_game()

    self.action_history.clear()
    self.action_history.append([])
    self.games_played += 1
    return self.gather_transition(autoreset=False)

  @episodic_log
  def calculate_reward(self):
    if self.simulation.game.ended:
      reward = self.game_value()
    else:
      reward = 0.0
      # board_mana_adv = sum((c.cost + 1 for c in self.simulation.player.characters)) - sum(
      #   (c.cost + 1 for c in self.simulation.opponent.characters))
      # reward = np.clip(board_mana_adv/10, -0.99, 0.99)
      # reward = (self.simulation.player.hero.health - self.simulation.opponent.hero.health) / self.starting_hp
    return np.array(reward, dtype=np.float32)

  @episodic_log
  def set_opponents(self, opponents: Iterable[agents.base_agent.Agent], opponent_obs_rmss=None):
    assert [isinstance(opponent, (agents.base_agent.Agent, agents.base_agent.Bot)) for opponent in opponents]
    assert [
      (opponent_obs_rms is None or isinstance(opponent_obs_rms, (baselines.common.running_mean_std.RunningMeanStd,)))
      for opponent_obs_rms in opponent_obs_rmss]

    self.opponents = opponents
    self.opponent_obs_rmss = opponent_obs_rmss

  @episodic_log
  def play_opponent_turn(self):
    assert self.simulation.game.current_player.controller.name == 'Opponent'
    while self.simulation.game.current_player.controller.name == 'Opponent':
      self.play_opponent_action()

  @episodic_log
  def play_opponent_action(self):
    assert self.simulation.game.current_player.controller.name == 'Opponent'
    observation, _, terminal, encoded_info = self.gather_transition(autoreset=False)

    if self.opponent_obs_rms is not None:
      observation = (observation - self.opponent_obs_rms.mean) / np.sqrt(self.opponent_obs_rms.var)

    observation = torch.FloatTensor(observation)
    observation = observation.unsqueeze(0)

    pa = encoded_info['possible_actions']
    pa = torch.FloatTensor(pa)
    encoded_info['possible_actions'] = pa.unsqueeze(0)
    # raise ValueError("Make sure/assert original info's order is correct!!!!")
    encoded_info['original_info'] = self.game_info()

    action = self.opponent.choose(observation, encoded_info)
    return self.step(action, autoreset=False)

  @episodic_log
  def step(self, encoded_action, autoreset=True):
    action = self.decode_action(int(encoded_action))

    self.action_history[-1].append(repr(action))
    if action.card is None:
      self.action_history.append([])

    assert isinstance(action, simulator.HSsimulation.Action)
    try:
      if encoded_action == 0:
        self.simulation.game.end_turn()

        if self.simulation.game.turn > 90:
          episodic_log(lambda a, b, c: None)(self, 'sudden_death', self.simulation.game.turn)
          self.simulation.sudden_death()
          raise GameOver

        if self.simulation.game.current_player.controller.name == 'Opponent':
          self.play_opponent_turn()
      else:
        self.simulation.step(action)
    except GameOver as e:
      if not self.simulation.game.ended:
        raise e
    return self.gather_transition(autoreset)

  @episodic_log
  def gather_transition(self, autoreset) -> Tuple[np.ndarray, np.ndarray, bool, Dict[Text, Any]]:
    possible_actions = self.simulation.actions()
    game_observation = self.simulation.observe()

    # board_adv = len(self.simulation.player.characters) - len(self.simulation.opponent.characters)
    # hand_adv = len(self.simulation.player.hand) - len(self.simulation.opponent.hand)
    # board_mana_adv = sum((c.cost for c in self.simulation.player.characters)) - sum(
    #   (c.cost for c in self.simulation.opponent.characters))
    # stats = np.array([board_adv, hand_adv, board_mana_adv])
    # game_observation = np.concatenate((game_observation, stats), axis=0)

    reward = self.calculate_reward()
    terminal = self.simulation.game.ended

    enc_possible_actions = self.encode_actions(possible_actions)

    possible_actions = utils.one_hot_actions((enc_possible_actions,), self.num_actions)

    history = []
    for turn in self.action_history:
      turn_history = ", ".join(reversed(turn))
      history.append(turn_history)

    info = {
      'possible_actions': possible_actions.squeeze(),
      'observation': game_observation,
      'reward': reward,
      'action_history': list(reversed(history)),
    }
    if terminal:
      info['game_statistics'] = {
        'num_games': self.games_played,
        'num_steps': self.episode_steps,
        'turn': self.simulation.game.turn,
        'outcome': reward,
      }
      if autoreset:
        new_obs, _, _, new_info = self.reset()
        game_observation = new_obs
        info['possible_actions'] = new_info['possible_actions']
        info['observation'] = new_info['observation']

    self.last_info = info
    return game_observation, reward, terminal, info

  @episodic_log
  def game_info(self):
    possible_actions = self.simulation.actions()
    info = {
      'possible_actions': possible_actions,
    }
    return info

  def cheat_hp(self, player_hp=None, opponent_hp=None):
    self.reset()
    MOONFIRE = "CS2_008"
    player = self.simulation.player
    opponent = self.simulation.opponent

    player_hp = player_hp or player.hero.health
    opponent_hp = opponent_hp or opponent.hero.health

    actor = self.simulation.game.current_player

    while player.hero.health != player_hp:
      actor.give(MOONFIRE).play(target=player.hero)

    while opponent.hero.health != opponent_hp:
      actor.give(MOONFIRE).play(target=opponent.hero)

    return self.gather_transition()

  def close(self):
    super(VanillaHS, self).close()
    if hasattr(self, '_gui'):
      del self._gui

  def __str__(self):
    return 'VanillaHS:{}'.format(self.opponents[-1].level, )
