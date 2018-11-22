import logging
import numpy as np

import fireplace
import fireplace.logging
import hearthstone
from fireplace.exceptions import GameOver
from fireplace.game import Game, PlayState
from gym import spaces

from environments import simulator
from environments import base_env
from shared import utils

from typing import Tuple, List


class VanillaHS(base_env.BaseEnv):
  class GameActions(object):
    PASS_TURN = (-1, -1)

  def __init__(self, max_cards_in_board=2, skip_mulligan=False,
    cheating_opponent=False, starting_hp=30):
    """
    A game with only vanilla monster cars, mage+warrior hero powers, 2 cards
    in front of each player.

    Args:
      starting_hp:
    """
    fireplace.cards.db.initialized = True
    print("Initializing card database")
    db, xml = hearthstone.cardxml.load()

    # allow coin, mage and warrior hero powers
    allowed_ids = ("GAME_005", "CS2_034", "CS2_102")
    for id, card in db.items():
      if card.description == "" or card.id in allowed_ids:
        fireplace.cards.db[id] = fireplace.cards.db.merge(id, card)

    logging.getLogger("fireplace").setLevel(logging.ERROR)

    simulator.HSsimulation._MAX_CARDS_IN_BOARD = max_cards_in_board
    self.max_cards_in_board = max_cards_in_board
    self.skip_mulligan = skip_mulligan
    self.lookup_action_id_to_obj = {}
    self.cheating_opponent = cheating_opponent
    self.starting_hp = starting_hp

    self.reinit_game()

  @property
  def action_space(self):
    # sources = self.simulation._MAX_CARDS_IN_BOARD + 1
    # targets = (self.simulation._MAX_CARDS_IN_BOARD + 1)
    return 2

  @property
  def observation_space(self):
    # 2 board of MAX_CARDS_IN_BOARD + hero, 2 stats per card
    return ((self.simulation._MAX_CARDS_IN_BOARD + 1) * 2) * 2

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

  def decode_action(self, encoded_action: Tuple[int, int]):
    assert isinstance(encoded_action, tuple)
    assert isinstance(encoded_action[0], (int, np.int64))

    return self.lookup_action_id_to_obj[encoded_action]

  @staticmethod
  def action_to_id(possible_action) -> Tuple[int, int]:
    card = possible_action.card
    if card is None:
      source_idx = -1
      target_idx = -1
    else:
      source_idx = card.zone_position
      target = possible_action.params['target']
      if target is None:
        target_idx = -1
      else:
        target_idx = target.zone_position

    # TODO: reverse lookup cache
    # action_id = self.action_to_ref.index((card_idx, target_idx))
    return source_idx, target_idx

  def reinit_game(self):
    self.simulation = simulator.HSsimulation(
      skip_mulligan=self.skip_mulligan,
      cheating_opponent=self.cheating_opponent,
      starting_hp=self.starting_hp,
    )
    cards = []
    for player in (self.simulation.player1, self.simulation.player2):
      last_card = player.hand[-1]
      if last_card.id == "GAME_005":  # "The Coin"
        last_card.discard()

      for card in player.deck:
        cards.append((card.atk, card.health))
      for card in player.hand:
        cards.append((card.atk, card.health))
    cards = np.array(cards)

    card_max = list(cards.max(axis=0))
    self.normalization_factors = np.array(
      card_max * self.max_cards_in_board + [1, 30] + card_max * self.max_cards_in_board + [1, 30]
    )
    self.games_played = 0
    self.games_finished = 0
    self.info = None
    self.actor_hero = None

  def game_value(self):
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
    return self.simulation._MAX_CARDS_IN_HAND

  def render(self, mode='human'):
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
    self.reinit_game()
    self.actor_hero = self.simulation.player.hero

    self.games_played += 1

    return self.gather_transition()

  def calculate_reward(self):
    if self.simulation.game.ended:
      reward = self.game_value()
    else:
      # reward = -0.01
      reward = 0.0
      # reward = -len(
      #  self.simulation.opponent.characters[1:]) / self.max_cards_in_board
      # reward /= 100
    return reward

  def set_opponent(self, opponent):
    self.opponent = opponent

  def play_opponent_turn(self):
    if self.opponent is None:
      with utils.suppress_stdout():
        fireplace.utils.play_turn(self.simulation.game)
    else:
      raise NotImplementedError
      while self.simulation.whosturn == opponent:
        self.play_opponent_action()

  def play_opponent_action(self):
    info = {'possible_actions': self.simulation.actions()}
    observation = self.simulation.observe()
    action = self.opponent.choose(observation, info)
    self.step(action)

  def step(self, encoded_action: Tuple[int, int]):
    action = self.decode_action(encoded_action)
    assert isinstance(action, simulator.HSsimulation.Action)
    try:
      if encoded_action == self.GameActions.PASS_TURN:
        self.simulation.game.end_turn()
        if self.simulation.game.current_player.controller.name == 'Opponent':
          self.play_opponent_turn()
      else:
        self.simulation.step(action)
    except GameOver as e:
      assert self.simulation.game.ended

    transition = self.gather_transition()
    return transition

  def gather_transition(self):
    possible_actions = self.simulation.actions()
    game_observation = self.simulation.observe()
    reward = self.calculate_reward()
    terminal = self.simulation.game.ended
    info = {
      'possible_actions': self.encode_actions(possible_actions),
      'original_info': {'possible_actions': possible_actions},
    }
    game_observation = game_observation / self.normalization_factors
    game_observation -= 0.5

    return game_observation, reward, terminal, info
