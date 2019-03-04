import hs_config

import logging
import gym.spaces
import pprint
import collections
import functools
import numpy as np

import fireplace
import fireplace.logging
import hearthstone
from fireplace.exceptions import GameOver
from fireplace.game import Game, PlayState

from environments import simulator
from environments import base_env
from shared import utils

from typing import Tuple, List, Dict, Text, Any
from baselines.common.running_mean_std import RunningMeanStd


class VanillaHS(base_env.BaseEnv):
  def __init__(
    self,
    max_cards_in_board=hs_config.VanillaHS.max_cards_in_board,
    max_cards_in_hand=hs_config.VanillaHS.max_cards_in_hand,
    skip_mulligan=True,
    cheating_opponent=False,
    starting_hp=hs_config.VanillaHS.starting_hp,
  ):
    """
    A game with only vanilla monster cars, mage+warrior hero powers, 2 cards
    in front of each player.

    Args:
      starting_hp:
    """
    self.id_to_action = [(-1, -1), ]
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
    if hs_config.VanillaHS.debug:
      self.log = collections.deque(maxlen=2)
      self.episodic_log('init', new_episode=True)
      logger = logging.getLogger('fireplace')
      logger.setLevel(logging.INFO)
    else:
      self.log = None
      logging.getLogger("fireplace").setLevel(logging.ERROR)

    # allow coin, mage and warrior hero powers
    allowed_ids = ("GAME_005", "CS2_034", "CS2_102")
    for id, card in db.items():
      if card.description == "" or card.id in allowed_ids:
        fireplace.cards.db[id] = fireplace.cards.db.merge(id, card)

    simulator.HSsimulation._MAX_CARDS_IN_BOARD = max_cards_in_board
    simulator.HSsimulation._MAX_CARDS_IN_HAND = max_cards_in_hand
    self.max_cards_in_board = max_cards_in_board

    self.skip_mulligan = skip_mulligan
    self.lookup_action_id_to_obj = {}
    self.cheating_opponent = cheating_opponent
    self.starting_hp = starting_hp

    self.ob_rms = None
    self.reinit_game()
    if hs_config.VanillaHS.normalize:
      self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)

  def episodic_log(self, *args, new_episode=False):
    if self.log is not None:
      if new_episode:
        self.log.append([])
      self.log[-1].append(tuple(args))

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

  @property
  def action_space(self):
    self.episodic_log('action_space')
    return gym.spaces.Discrete(self.num_actions)

  @property
  def observation_space(self):
    # 2 board of MAX_CARDS_IN_BOARD + hero, 2 stats per card
    self.episodic_log('observation_space')
    obs, _, _, _ = self.gather_transition()
    return gym.spaces.Box(low=-1, high=10, shape=obs.shape, dtype=obs.dtype)

  def encode_actions(self, actions: Tuple[simulator.HSsimulation.Action]):
    assert isinstance(actions, tuple)
    assert isinstance(actions[0], simulator.HSsimulation.Action)
    self.episodic_log('encode_actions', actions)

    self.lookup_action_id_to_obj.clear()
    encoded_actions = []
    for action in actions:
      action_id = self.action_to_id(action)
      self.lookup_action_id_to_obj[action_id] = action
      encoded_actions.append(action_id)
    self.episodic_log('> encode_actions', tuple(encoded_actions))
    return tuple(encoded_actions)

  def decode_action(self, encoded_action: int):
    assert isinstance(encoded_action, (int, np.int64))
    self.episodic_log('decode_action', encoded_action)
    return self.lookup_action_id_to_obj[encoded_action]

  def action_to_id(self, possible_action) -> Tuple:
    self.episodic_log('action_to_id', possible_action)

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
      return self.action_to_id_dict[(source_idx, target_idx)]
    except KeyError as e:
      print(possible_action)
      raise e

  def reinit_game(self):
    self.episodic_log('reinit_game', new_episode=True)

    self.simulation = simulator.HSsimulation(
      skip_mulligan=self.skip_mulligan,
      cheating_opponent=self.cheating_opponent,
      starting_hp=self.starting_hp,
    )
    cards = []
    for player in (self.simulation.player1, self.simulation.player2):
      for c in player.hand:
        if c.id == "GAME_005":  # "The Coin"
          c.discard()

      for card in player.deck:
        cards.append((card.atk, card.health))
      for card in player.hand:
        cards.append((card.atk, card.health))
    cards = np.array(cards)

    card_max = list(cards.max(axis=0) + [1])

    self.games_played = 0
    self.episode_steps = 0
    self.games_finished = 0
    self.info = None
    self.actor_hero = None

  def game_value(self):
    self.episodic_log('game_value')
    player = self.simulation.player
    if player.playstate == PlayState.WON:
      return +1.0
    elif player.playstate == PlayState.LOST:
      return -1.0
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
  def cards_in_hand(self):
    self.episodic_log('cards_in_hand')
    return self.simulation._MAX_CARDS_IN_HAND

  def render(self, mode='human', info={}):
    board = ""
    board += "-" * 100
    board += "\nYOU:{player_hp}\n[{player_hand}]\n{player_board}\n{separator}\nboard:{o_board}\nENEMY:{o_hp}\n[{o_hand}]".format(
      player_hp=self.simulation.player.hero.health,
      player_hand="\t".join(c.data.name for c in self.simulation.player.hand),
      player_board="\t \t".join(
        c.data.name for c in self.simulation.player.characters[1:]),
      separator="_" * 100,
      o_hp=self.simulation.opponent.hero.health,
      o_hand="\t".join(
        c.data.name for c in self.simulation.player.characters[1:]),
      o_board="\t \t".join(
        c.data.name for c in self.simulation.opponent.characters[1:]),
    )
    board += "\n" + "*" * 100 + "\n" * 3
    return board

  def reset(self):
    self.episodic_log('reset')
    self.reinit_game()
    self.actor_hero = self.simulation.player.hero

    self.games_played += 1
    self.episode_steps = 0
    return self.gather_transition()

  def calculate_reward(self):
    self.episodic_log('calculate_reward')
    if self.simulation.game.ended:
      reward = self.game_value()
    else:
      reward = 0.0
      # board_mana_adv = sum((c.cost + 1 for c in self.simulation.player.characters)) - sum(
      #   (c.cost + 1 for c in self.simulation.opponent.characters))
      # reward = np.clip(board_mana_adv/10, -0.99, 0.99)
      # reward = (self.simulation.player.hero.health - self.simulation.opponent.hero.health) / self.starting_hp
    return np.array(reward, dtype=np.float32)

  def set_opponent(self, opponent):
    self.episodic_log('set_opponent')
    self.opponent = opponent

  def play_opponent_turn(self):
    self.episodic_log('play_turn')
    assert self.simulation.game.current_player.controller.name == 'Opponent'
    while self.simulation.game.current_player.controller.name == 'Opponent':
      self.play_opponent_action()

  def play_opponent_action(self):
    self.episodic_log('play_opponent_action')
    assert self.simulation.game.current_player.controller.name == 'Opponent'
    observation, _, terminal, info = self.gather_transition(pickle=False)
    action = self.opponent.choose(observation, info)
    self.step(action)
    trans = self.gather_transition(pickle=False)
    return trans

  def step(self, encoded_action):
    self.episodic_log('step', encoded_action, self.simulation.game.turn)
    self.episode_steps += 1
    action = self.decode_action(int(encoded_action))
    assert isinstance(action, simulator.HSsimulation.Action)
    try:
      if encoded_action == 0:
        self.simulation.game.end_turn()

        if self.simulation.game.turn > 90:
          self.episodic_log('sudden death', self.simulation.game.turn)
          # print('SUDDEN DEATH')
          self.simulation.sudden_death()
          raise GameOver

        if self.simulation.game.current_player.controller.name == 'Opponent':
          self.play_opponent_turn()
      else:
        self.simulation.step(action)
    except GameOver as e:
      assert self.simulation.game.ended

    game_observation, reward, terminal, info = self.gather_transition()

    if terminal:
      info['game_statistics'] = {
        'num_games': self.games_played,
        'num_steps': self.episode_steps,
        'turn': self.simulation.game.turn,
      }
      self.reset()
    return game_observation, reward, terminal, info

  def gather_transition(self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[Text, Any]]:
    self.episodic_log('gather_transition')
    possible_actions = self.simulation.actions()
    game_observation = self.simulation.observe()

    # board_adv = len(self.simulation.player.characters) - len(self.simulation.opponent.characters)
    # hand_adv = len(self.simulation.player.hand) - len(self.simulation.opponent.hand)
    # board_mana_adv = sum((c.cost for c in self.simulation.player.characters)) - sum(
    #   (c.cost for c in self.simulation.opponent.characters))
    # stats = np.array([board_adv, hand_adv, board_mana_adv])
    # game_observation = np.concatenate((game_observation, stats), axis=0)

    if self.ob_rms is not None:
      self.ob_rms.update(game_observation)
      game_observation = (game_observation - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8)

    reward = self.calculate_reward()
    terminal = self.simulation.game.ended
    enc_possible_actions = self.encode_actions(possible_actions)

    possible_actions = utils.one_hot_actions((enc_possible_actions,), self.num_actions)
    info = {
      'possible_actions': possible_actions.squeeze()
    }
    return game_observation, reward, terminal, info

  def game_info(self):
    self.episodic_log('game_info')
    possible_actions = self.simulation.actions()
    info = {
      'possible_actions': self.encode_actions(possible_actions),
      'original_info': {'possible_actions': possible_actions},
    }
    return info
