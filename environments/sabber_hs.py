import random
import warnings
from enum import IntEnum
from typing import Tuple, Dict, Text, Any

import grpc
import gym
import numpy as np
import torch

import environments.base_env
import hs_config
import sb_env.SabberStone_python_client.python_pb2 as python_pb2
import sb_env.SabberStone_python_client.python_pb2_grpc as python_pb2_grpc
import shared.env_utils
import specs

_MAX_CARDS_IN_HAND = 10
_MAX_CARDS_IN_PLAYER_BOARD = 7 + 1  # opponent face
_MAX_CARDS_IN_BOARD = _MAX_CARDS_IN_PLAYER_BOARD * 2
_MAX_TYPE = 7
_ACTION_SPACE = 249
_STATE_SPACE = 92  # state space includes card_index


class PlayerTaskType(IntEnum):
  CHOOSE = 0
  CONCEDE = 1
  END_TURN = 2
  HERO_ATTACK = 3
  HERO_POWER = 4
  MINION_ATTACK = 5
  PLAY_CARD = 6


# class BaseEnv(gym.Env, ABC):
#   class GameActions(Enum):
#     PASS_TURN = 0
#
#   @property
#   @abstractmethod
#   def cards_in_hand(self):
#     raise NotImplemented
#
#   @abstractmethod
#   def play_opponent_action(self):
#     raise NotImplemented
#
#   @abstractmethod
#   def game_value(self):
#     raise NotImplemented
#
#   @abstractmethod
#   def set_opponents(self, opponents, opponent_obs_rmss):
#     raise NotImplemented
#

def parse_hero(hero):
  # TODO change damage in health
  return np.array([hero.atk, hero.damage, hero.exhausted, hero.power.exhausted])


# TODO add card repr
def parse_card(card):
  return np.array([card.card_id, card.atk, card.base_health, card.cost])


def parse_minion(card):
  return np.array([card.atk, card.damage, card.exhausted])


def pad(x, shape, parse_card):
  v = np.zeros(shape)
  if len(x) == 0:
    return v
  try:
    _v = np.vstack(list(map(parse_card, x)))
    v[:len(_v)] = _v
  except ValueError as e:
    raise e
  return v


def parse_player(player):
  hero = parse_hero(player.hero)
  # TODO parse spells and minion differently
  hand_zone = pad(player.hand_zone.entities, shape=(_MAX_CARDS_IN_HAND, 4), parse_card=parse_card)
  board_zone = pad(player.board_zone.minions, shape=(_MAX_CARDS_IN_PLAYER_BOARD - 1, 3), parse_card=parse_minion)
  return hero, board_zone.flatten(), hand_zone.flatten()


def build_state(game):
  mana_p1 = np.array((game.CurrentPlayer.remaining_mana,))
  mana_p2 = np.array((game.CurrentOpponent.remaining_mana,))  # TODO or base mana
  player1 = np.hstack(parse_player(game.CurrentPlayer))
  hero_p2, board_p2, _ = parse_player(game.CurrentOpponent)
  return np.hstack([mana_p1, player1, mana_p2, hero_p2, board_p2])


def full_random_game(stub, deck1, deck2):
  game = stub.NewGame(python_pb2.DeckStrings(deck1=deck1, deck2=deck2))
  options_list = []
  while game.state != python_pb2.Game.State.COMPLETE:
    options = stub.Options(game)
    for option in options:
      options_list.append(option)
    option = options_list[random.randrange(len(options_list))]
    # returns the updated game state
    game = stub.Process(option)
    options_list.clear()


def connect(addr='localhost:50052'):
  try:
    channel = grpc.insecure_channel(addr)
    stub = python_pb2_grpc.SabberStonePythonStub(channel)
    print(f"Connected to {addr}")
    return stub
  except Exception as e:
    raise e


# check for zone
def enumerate_actions():
  id_to_action = [(PlayerTaskType.END_TURN, 0, 0)]

  # place minions
  for src_id in range(_MAX_CARDS_IN_HAND):
    for target_id in range(_MAX_CARDS_IN_BOARD):
      id_to_action.append((PlayerTaskType.PLAY_CARD, src_id, target_id))

  # attack
  for src_id in range(_MAX_CARDS_IN_PLAYER_BOARD):
    for target_id in range(_MAX_CARDS_IN_PLAYER_BOARD, _MAX_CARDS_IN_PLAYER_BOARD * 2):
      id_to_action.append((PlayerTaskType.MINION_ATTACK, src_id, target_id))

  # hero power`
  for target_id in range(_MAX_CARDS_IN_BOARD):
    id_to_action.append((PlayerTaskType.HERO_POWER, 0, target_id))

  # hero attack
  for target_id in range(_MAX_CARDS_IN_PLAYER_BOARD, _MAX_CARDS_IN_PLAYER_BOARD * 2):
    id_to_action.append((PlayerTaskType.HERO_ATTACK, 0, target_id))

  action_to_id_dict = {v: k for k, v in enumerate(id_to_action)}
  assert len(id_to_action) == _ACTION_SPACE
  return action_to_id_dict


# class Stub:
#   stub = connect()
#
#   def __init__(self):
#     pass
#
#   def __getattribute__(self, name):
#     attr = object.__getattribute__(Stub.stub, name)
#     if hasattr(attr, '__call__'):
#       def newfunc(*args, **kwargs):
#         print('>calling', name)
#         result = attr(*args, **kwargs)
#         print('<<<<<< calling', name)
#         return result
#
#       return newfunc
#     else:
#       return attr
# stub = Stub()


class Sabbertsone(environments.base_env.BaseEnv):
  DECK1 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="
  DECK2 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="
  stub = connect()

  def __init__(self, seed: int = None, extra_seed: int = None, host: Text = "localhost:5002"):
    if seed or extra_seed:
      warnings.warn("Setting the seed is not implemented")

    self.channel = host
    self.game = Sabbertsone.stub.NewGame(python_pb2.DeckStrings(deck1=self.DECK1, deck2=self.DECK2))

    self.action_space = gym.spaces.Discrete(n=_ACTION_SPACE)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(_STATE_SPACE,), dtype=np.int)

    self.opponent = None
    self.opponents = [None, ]

    self.opponent_obs_rms = None
    self.opponent_obs_rmss = [None, ]

    # self.games_played = 0
    # self.episode_steps = 0
    # self.games_finished = 0
    # self.info = None
    self.enumerate_actions = enumerate_actions()

  def cards_in_hand(self):
    raise len(self.game.player1.hand_zone)

  def game_value(self):
    if self.game.CurrentPlayer == 1 and self.game.CurrentPlayer.play_state == python_pb2.Controller.PlayState.WON:
      reward = 1
    elif self.game.CurrentPlayer.play_state == python_pb2.Controller.LOST:
      reward = -1
    else:
      reward = 0
    return np.array(reward, dtype=np.float32)

  def is_terminal(self):
    return self.game.state == python_pb2.Game.COMPLETE

  def original_info(self):
    possible_options = Sabbertsone.stub.GetOptions(self.game).list
    return {"possible_actions": tuple(possible_options)}

  @classmethod
  def parse_options(cls, game, option_id_to_dict):
    options = cls.stub.GetOptions(game.id).list
    possible_options = {}
    for option in options:
      # removing chooses taks
      if option.type == PlayerTaskType.CHOOSE:
        raise NotImplementedError
      action = option_id_to_dict[(option.type, option.source_position, option.target_position)]
      possible_options[action] = option
    assert len(possible_options) > 0  # pass should always be there
    return possible_options

  @shared.env_utils.episodic_log
  def step(self, action_id: np.ndarray, auto_reset: bool = False):
    assert specs.check_positive_type(action_id, (np.ndarray, int), strict=False)
    assert isinstance(action_id, int) or (action_id.dtype == np.int64 and action_id.size == 1)
    action_id = int(action_id)

    actions = Sabbertsone.parse_options(self.game, self.enumerate_actions)
    selected_action = actions[action_id]
    self.game = Sabbertsone.stub.Process(selected_action)

    if self.game.turn > hs_config.Environment.max_turns:
      state, reward, done, info = self.gather_transition(auto_reset=auto_reset)
      done = True
      reward = 0.
      return state, reward, done, info

    if self.game.CurrentPlayer.id == 2:
      self.play_opponent_turn()

    return self.gather_transition(auto_reset=auto_reset)

  @shared.env_utils.episodic_log
  def reset(self):
    self.game = Sabbertsone.stub.Reset(self.game.id)

    self._sample_opponent()
    self.episode_steps = 0
    self.info = None

    if self.game.CurrentPlayer.id == 2:
      self.play_opponent_turn()

    return self.gather_transition(auto_reset=False)

  @shared.env_utils.episodic_log
  def gather_transition(self, auto_reset: bool) -> Tuple[np.ndarray, np.ndarray, bool, Dict[Text, Any]]:
    done = self.is_terminal()
    reward = self.game_value()
    state = build_state(self.game)

    actions = Sabbertsone.parse_options(self.game, self.enumerate_actions)
    possible_actions = np.zeros(_ACTION_SPACE, dtype=np.float32)
    possible_actions[list(actions.keys())] = 1

    info = {
      'possible_actions': possible_actions,
      'action_history': [],
      'observation': state,
      'reward': reward
    }

    if done:
      info['game_statistics'] = {
        # 'num_games': self.games_played,
        # 'num_steps': self.episode_steps,
        # 'turn': self.game.turn,
        'outcome': reward,
      }
      warnings.warn("autoreset should be in step not in gather_tansition")
      if auto_reset:
        print('AUTORESET')
        state, _, _, info = self.reset()
        info['observation'] = state
        info['reward'] = reward

    assert info['possible_actions'].max() == 1
    return state, reward, done, info

  def _sample_opponent(self):
    if np.random.uniform() < 0.2 or self.opponent is None:
      # off by one?
      k = np.random.randint(0, high=len(self.opponents))
      self.opponent = self.opponents[k]
      try:
        self.opponent_obs_rmss = self.opponent_obs_rmss[k]
      except TypeError:
        pass
    #   self.games_played = 0
    # else:
    #   self.games_played += 1

  def set_opponents(self, opponents, opponent_obs_rmss=None):
    # if not isinstance(opponents, list):
    #   opponents = [opponents]
    self.opponents = opponents
    self.opponent_obs_rmss = opponent_obs_rmss

  def play_opponent_action(self):
    observation, _, terminal, info = self.gather_transition(auto_reset=False)

    if self.opponent_obs_rms is not None:
      observation = (observation - self.opponent_obs_rms.mean) / np.sqrt(self.opponent_obs_rms.var)

    observation = torch.FloatTensor(observation)
    observation = observation.unsqueeze(0)

    pa = info['possible_actions']
    pa = torch.FloatTensor(pa)
    info['possible_actions'] = pa.unsqueeze(0)
    # raise ValueError("Make sure/assert original info's order is correct!!!!")
    info['original_info'] = self.original_info()

    action = self.opponent.choose(observation, info)

    return self.step(action, auto_reset=False)

  def play_opponent_turn(self):
    assert self.game.CurrentPlayer.id == 2
    while self.game.CurrentPlayer.id == 2:
      self.play_opponent_action()

  def close(self):
    python_pb2_grpc.ServerHandleStub(channel=self.channel).Close()
