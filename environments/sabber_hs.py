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

AGENT_ID = 1
OPPONENT_ID = 2


class PlayerTaskType(IntEnum):
  CHOOSE = 0
  CONCEDE = 1
  END_TURN = 2
  HERO_ATTACK = 3
  HERO_POWER = 4
  MINION_ATTACK = 5
  PLAY_CARD = 6


# TODO add card repr
def parse_card(card):
  return np.array((card.card_id, card.atk, card.base_health, card.cost))

  # TODO parse spells and minion differently
def parse_minion(card):
  return np.array([card.atk, card.base_health - card.damage, card.exhausted])


def pad(x, length, parse_card):
  v = -np.ones(length)
  if x:
    _v = np.hstack([parse_card(xi) for xi in x])
    v[:_v.shape[0]] = _v
  return v


def parse_player(player):
  hero = np.array(
    [player.hero.atk, player.hero.base_health - player.hero.damage, player.hero.exhausted, player.hero.power.exhausted])
  hand_zone = pad(player.hand_zone.entities, length=_MAX_CARDS_IN_HAND * 4, parse_card=parse_card)
  board_zone = pad(player.board_zone.minions, length=(_MAX_CARDS_IN_PLAYER_BOARD - 1) * 3, parse_card=parse_minion)
  return hero, board_zone, hand_zone


def build_state(game):
  hero_p2, board_p2, _ = parse_player(game.CurrentOpponent)
  return np.hstack((
    game.CurrentPlayer.remaining_mana,
    *parse_player(game.CurrentPlayer),
    game.CurrentOpponent.remaining_mana,
    *hero_p2,
    *board_p2
  ))


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
  address = "localhost:50052"
  channel = grpc.insecure_channel(address)
  stub = python_pb2_grpc.SabberStonePythonStub(channel)

  def __init__(self, seed: int = None, extra_seed: int = None):
    super().__init__()
    if seed is not None or extra_seed is not None:
      warnings.warn("Setting the seed is not implemented")

    self.game_ref = Sabbertsone.stub.NewGame(python_pb2.DeckStrings(deck1=self.DECK1, deck2=self.DECK2))

    self.action_space = gym.spaces.Discrete(n=_ACTION_SPACE)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(_STATE_SPACE,), dtype=np.int)

    self.action_to_id = enumerate_actions()

  def cards_in_hand(self):
    raise len(self.game_ref.player1.hand_zone)

  def game_value(self):
    player = self.game_ref.CurrentPlayer

    if player.play_state == python_pb2.Controller.WON:  # maybe PlayState
      reward = 1
    elif player.play_state == python_pb2.Controller.LOST:
      reward = -1
    else:
      reward = 0

    warnings.warn('check if opponent wins/loses the reward goes to the agent')
    return np.array(reward, dtype=np.float32)

  def is_terminal(self):
    return terminal

  def original_info(self):
    possible_options = Sabbertsone.stub.GetOptions(self.game_ref).list
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
  def step(self, action_id: np.ndarray, auto_reset: bool = True):
    assert specs.check_positive_type(action_id, (np.ndarray, int), strict=False)
    assert isinstance(action_id, int) or (action_id.dtype == np.int64 and action_id.size == 1)
    action_id = int(action_id)

    actions = Sabbertsone.parse_options(self.game_ref, self.action_to_id)
    selected_action = actions[action_id]
    self.game_ref = Sabbertsone.stub.Process(selected_action)

    if self.game_ref.turn > hs_config.Environment.max_turns:
      state, reward, done, info = self.gather_transition(auto_reset=auto_reset)
      done = True
      reward = 0.
      return state, reward, done, info

    if self.game_ref.CurrentPlayer.id == 2:
      self.play_opponent_turn()

    return self.gather_transition(auto_reset=auto_reset)

  @shared.env_utils.episodic_log
  def reset(self):
    self.game_ref = Sabbertsone.stub.Reset(self.game_ref.id)
    self._sample_opponent()
    # self.episode_steps = 0
    # self.info = None
    if self.game_ref.CurrentPlayer.id == 2:
      self.play_opponent_turn()
    return self.gather_transition(auto_reset=False)

  @shared.env_utils.episodic_log
  def gather_transition(self, auto_reset: bool) -> Tuple[np.ndarray, np.ndarray, bool, Dict[Text, Any]]:
    assert not auto_reset or self.game_ref.CurrentPlayer.id == AGENT_ID  # the opponent cannot auto_reset

    terminal = self.game_ref.state == python_pb2.Game.COMPLETE
    reward = self.game_value()
    state = build_state(self.game_ref)

    possible_actions = np.zeros(_ACTION_SPACE, dtype=np.float32)
    if not terminal:
      actions = Sabbertsone.parse_options(self.game_ref, self.action_to_id)
      possible_actions[list(actions.keys())] = 1

    if terminal:
      if auto_reset:
        state, _, _, info = self.reset()
      else:
        raise self.GameOver

    info = {
      'observation': state,
      'reward': reward,
      'possible_actions': possible_actions,
      'action_history': [],
    }

    assert info['possible_actions'].max() == 1
    return state, reward, terminal, info

  def _sample_opponent(self):
    pick = np.random.uniform()
    if pick < 0.5:
      k = -1
    else:
      k = random.randint(0, len(self.opponents) - 1)

    self.opponent = self.opponents[k]
    if self.opponent_obs_rmss is not None:
      self.opponent_obs_rmss = self.opponent_obs_rmss[k]

  def play_opponent_action(self):
    observation, _, terminal, info = self.gather_transition(auto_reset=False)

    if self.opponent_obs_rms is not None:
      observation = (observation - self.opponent_obs_rms.mean) / np.sqrt(self.opponent_obs_rms.var)

    observation = torch.FloatTensor(observation)
    observation = observation.unsqueeze(0)

    pa = info['possible_actions']
    pa = torch.FloatTensor(pa)
    info['possible_actions'] = pa.unsqueeze(0)
    info['original_info'] = self.original_info()

    action = self.opponent.choose(observation, info)
    return self.step(action, auto_reset=False)

  def play_opponent_turn(self):
    assert self.game_ref.CurrentPlayer.id == 2
    try:
      while self.game_ref.CurrentPlayer.id == 2:
        self.play_opponent_action()
    except self.GameOver:
      pass

  def close(self):
    # Not sure about this.
    python_pb2_grpc.ServerHandleStub(channel=Sabbertsone.channel).Close()
