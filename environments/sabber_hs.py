import warnings
from typing import Tuple, Dict, Text, Any

import grpc
import gym
import numpy as np
import torch

import environments.base_env
import hs_config
import sb_env.SabberStone_python_client.python_pb2 as python_pb2
import sb_env.SabberStone_python_client.python_pb2_grpc as python_pb2_grpc
import shared.constants as C
import shared.env_utils
import shared.utils
from shared.constants import PlayerTaskType, BoardPosition, HandPosition, GameStatistics, _ACTION_SPACE, _STATE_SPACE


def parse_hero(hero):
  return np.array([hero.atk, hero.base_health - hero.damage, hero.exhausted, hero.power.exhausted])


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
  hand_zone = pad(player.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse_card=parse_card)
  board_zone = pad(player.board_zone.minions, length=hs_config.Environment.max_cards_in_board * 3,
                   parse_card=parse_minion)
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


def enumerate_actions():
  id_to_action = [(PlayerTaskType.END_TURN, 0, 0)]

  # place minions or play spell
  for src_id in range(hs_config.Environment.max_cards_in_hand):
    for target_id in range(hs_config.Environment.max_entities_in_board * 2):
      id_to_action.append((PlayerTaskType.PLAY_CARD, HandPosition(src_id), BoardPosition(target_id)))

  # attack
  for src_id in range(hs_config.Environment.max_entities_in_board):
    for target_id in range(hs_config.Environment.max_entities_in_board,
                           hs_config.Environment.max_entities_in_board * 2):
      id_to_action.append((PlayerTaskType.MINION_ATTACK, BoardPosition(src_id), BoardPosition(target_id)))

  # hero power`
  for target_id in range(hs_config.Environment.max_entities_in_board * 2):
    id_to_action.append((PlayerTaskType.HERO_POWER, BoardPosition(0), BoardPosition(target_id)))

  # hero attack
  for target_id in range(hs_config.Environment.max_entities_in_board, hs_config.Environment.max_entities_in_board * 2):
    id_to_action.append((PlayerTaskType.HERO_ATTACK, BoardPosition(0), BoardPosition(target_id)))

  action_to_id_dict = {v: k for k, v in enumerate(id_to_action)}
  assert len(id_to_action) == _ACTION_SPACE
  return action_to_id_dict


def game_stats(game):
  player = game.CurrentPlayer
  opponent = game.CurrentOpponent

  power = [(minion.atk, minion.base_health) for minion in player.board_zone.minions]
  if len(power):
    power, value = np.sum(power, axis=0)
  else:
    power = 0
    value = 0
  defense = sum([minion.base_health for minion in opponent.board_zone.minions])

  opponent_life = opponent.hero.base_health - opponent.hero.damage
  hero_life = player.hero.base_health - player.hero.damage

  n_remaining_turns = power / opponent_life

  mana_adv = (player.base_mana - player.remaining_mana)
  hand_adv = (len(player.hand_zone.entities) - len(opponent.hand_zone.entities))
  draw_adv = (len(player.deck_zone.entities) - len(opponent.deck_zone.entities))  # number of remaining cards
  life_adv = opponent_life - hero_life
  minion_adv = value - defense

  return GameStatistics(mana_adv, hand_adv, draw_adv, life_adv, n_remaining_turns, minion_adv)
  # return {'mana_adv': mana_adv, 'hand_adv': hand_adv, 'draw_adv': draw_adv, 'life_adv': life_dav,
  #         'n_turns_left': n_remaining_turns, 'minion_adv': minion_adv}


def random_subset(opponents: list, k: int) -> tuple:
  assert len(opponents) > k
  n = 0
  result = []

  for idx, opponent in enumerate(opponents):
    n += 1
    if len(result) < k:
      result.append((opponent, idx))
    else:
      s = int(np.random.uniform() * n)
      if s < k:
        result[s] = (opponent, idx)
  return result[0]

from shared.utils import Timer # TODO check this import

class Sabbertsone(environments.base_env.RenderableEnv):
  DECK1 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="
  DECK2 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="

  action_to_id = enumerate_actions()

  hand_encoding_size = 4  # atk, health, exhaust
  hero_encoding_size = 4  # atk, health, exhaust, hero_power
  minion_encoding_size = 3  # atk, health, exhaust
  board_to_board = {PlayerTaskType.MINION_ATTACK, PlayerTaskType.HERO_ATTACK, PlayerTaskType.HERO_POWER}

  def __init__(self, address: str, seed: int = None, extra_seed: int = None):
    super().__init__()
    self.gui = None
    self.logger = Timer(__class__.__name__, id=extra_seed, verbosity=hs_config.verbosity)
    self.extra_seed = extra_seed
    if seed is not None or extra_seed is not None:
      self.logger.warning("Setting the seed is not implemented")

    with self.logger("call_init"):
      self.channel = grpc.insecure_channel(address)
      self.stub = python_pb2_grpc.SabberStonePythonStub(self.channel)
      self.game_ref = self.stub.NewGame(python_pb2.DeckStrings(deck1=Sabbertsone.DECK1, deck2=Sabbertsone.DECK2))

    self.action_space = gym.spaces.Discrete(n=_ACTION_SPACE)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(_STATE_SPACE,), dtype=np.int)
    self.turn_stats = []
    self._game_matrix = {}
    self.logger.info(f"Env with id {extra_seed} started.")

  def cards_in_hand(self):
    raise len(self.game_ref.player1.hand_zone)

  def game_value(self):
    player = self.game_ref.CurrentPlayer
    if player.play_state == python_pb2.Controller.WON:  # maybe PlayState
      reward = 1
    elif player.play_state in (python_pb2.Controller.LOST, python_pb2.Controller.TIED):
      reward = -1
    else:
      reward = 0
    if self.game_ref.state == python_pb2.Game.COMPLETE:
      reward = reward if player.id == C.AGENT_ID else -reward
    return np.array(reward, dtype=np.float32)

  def original_info(self):
    warnings.warn("minimize stub calls (superslow), options can be reused here")
    # possible_options = Sabbertsone.stub.GetOptions(self.game_ref).list
    return {
      "observation": self.game_ref,
      "possible_actions": self.parse_options(self.game_ref),
    }

  def parse_options(self, game, return_options=False):
    options = self.stub.GetOptions(game.id).list
    possible_options = {}

    for option in options:
      option_type = option.type
      assert option_type != PlayerTaskType.CHOOSE

      if option_type == PlayerTaskType.END_TURN:
        action_id = self.GameActions.PASS_TURN
      else:
        source = option.source_position
        target = option.target_position

        assert (option_type not in (PlayerTaskType.HERO_ATTACK, PlayerTaskType.HERO_POWER)
                or source == BoardPosition.Hero)

        action_hash = (option_type, source, target)
        action_id = self.action_to_id[action_hash]

      assert action_id not in possible_options
      possible_options[action_id] = option

    assert len(possible_options) > 0  # pass should always be there

    if return_options:
      return possible_options, options
    else:
      return possible_options

  def update_stats(self):
    if self.game_ref.CurrentPlayer.id == 1:
      new_stats = game_stats(self.game_ref)
      self.turn_stats.append(new_stats)

  @shared.env_utils.episodic_log
  def step(self, action_id: np.ndarray, auto_reset: bool = True):
    assert hasattr(action_id, '__int__')
    action_id = int(action_id)

    self.logger.info(self)

    try:
      with self.logger("parse_options"):
        actions = self.parse_options(self.game_ref)

      selected_action = actions[action_id]

      with self.logger("update_stats"):
        self.update_stats()

      assert self.game_ref.state != python_pb2.Game.COMPLETE

      with self.logger("call_process"):
        self.game_ref = self.stub.Process(selected_action)

      if self.game_ref.turn > hs_config.Environment.max_turns:
        state, reward, done, info = self.gather_transition(auto_reset=auto_reset)
        done = True
        reward = 0.
        return state, reward, done, info

      if self.game_ref.CurrentPlayer.id == 2:  # and self.game_ref.state != python_pb2.Game.COMPLETE:
        self.play_opponent_turn()

    except self.GameOver:
      assert self.game_ref.state == python_pb2.Game.COMPLETE
      self.logger.error(f"GameOver called from player {self.game_ref.CurrentPlayer.id}")

    return self.gather_transition(auto_reset=auto_reset)

  @shared.env_utils.episodic_log
  def reset(self):
    self.logger.info(f"Reset called from player {self.game_ref.CurrentPlayer.id}")
    with self.logger("call_reset"):
      self.game_ref = self.stub.Reset(self.game_ref.id)
    self._sample_opponent()
    # self.turn_stats = []
    # self.episode_steps = 0
    # self.info = None
    if self.game_ref.CurrentPlayer.id == C.OPPONENT_ID:
      self.play_opponent_turn()
    return self.gather_transition(auto_reset=False)

  @shared.env_utils.episodic_log
  def _gather_transition(self, auto_reset: bool) -> Tuple[np.ndarray, np.ndarray, bool, Dict[Text, Any]]:
    assert shared.utils.can_autoreset(auto_reset, self.game_ref) or self.game_ref.turn > hs_config.Environment.max_turns

    terminal = self.game_ref.state == python_pb2.Game.COMPLETE
    assert self.game_ref.state in (
      python_pb2.Game.INVALID, python_pb2.Game.LOADING, python_pb2.Game.RUNNING, python_pb2.Game.COMPLETE,)

    with self.logger("get_value"):
      reward = self.game_value()

    with self.logger("build_state"):
      state = build_state(self.game_ref)

    possible_actions = np.zeros(_ACTION_SPACE, dtype=np.float32)
    if not terminal:
      actions = self.parse_options(self.game_ref)
      possible_actions[list(actions.keys())] = 1

    info = {
      'observation': state,
      'reward': reward,
      'possible_actions': possible_actions,
      'action_history': [],
      'game_statistics': {}
    }
    if terminal:

      if auto_reset:
        # TODO maybe make me better
        self.game_matrix(self.current_k, reward)

        game_stats = GameStatistics(*zip(*self.turn_stats))
        game_stats = {'avg_' + k:v for k, v in zip(GameStatistics._fields, np.mean(game_stats, axis=1))}
        game_stats['outcome'] = reward
        game_stats['life_adv'] = self.turn_stats[-1].life_adv
        counts = np.array([v[1] for v in self._game_matrix.values()])
        game_stats['opponent_var'] = counts.var()
        game_stats['opponent_mean'] = counts.mean()

        self.logger.log_stats(game_stats)

        with self.logger("reset_env"):
          state, _, _, _info = self.reset()
        info['observation'] = state
        info['possible_actions'] = _info['possible_actions']
      else:
        raise self.GameOver

    assert info['possible_actions'].max() == 1 or terminal
    return state, reward, terminal, info

  def game_matrix(self, idx, reward):
    try:
      n, score = self._game_matrix[idx]
    except KeyError:
      n = 0
      score = 0
    self._game_matrix[idx] = (n + 1, score + reward)

  def print_nash(self):
    # t = lambda y, x: f"player {y}: played {x[0]}, score {x[1]}"
    # print("\n".join(list(map(t, self._game_matrix.items()))))
    for k, v in self._game_matrix.items():
      print(f"[ENV NASH]\t agent:{k}, N:{v[0]}, score:{v[1]}")

  def _sample_opponent(self):
    p = np.ones(shape=(len(self.opponents)))

    if len(self.opponents) > 1:
      if len(self._game_matrix.values()) > 1:
        counts = [v[0] for v in self._game_matrix.values()]
        idxs = list(self._game_matrix.keys())
        counts = 1 / np.array(counts)
        p[idxs] = counts
      assert p.sum() > 0
      p /= p.sum()

    k = np.random.choice(np.arange(0, len(self.opponents)), p=p)
    self.logger.info(f"Sampled new opponent with id {k} and prob {p[k]}")
    self.opponent = self.opponents[k]

    if self.opponent_obs_rmss is not None:
      self.opponent_obs_rmss = self.opponent_obs_rmss[k]

    self.current_k = k

  def play_opponent_action(self):
    assert self.game_ref.CurrentPlayer.id == C.OPPONENT_ID
    with self.logger("opponent_step"):
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
    assert self.game_ref.CurrentPlayer.id == C.OPPONENT_ID
    return self.step(action, auto_reset=False)

  def play_opponent_turn(self):
    for _ in range(1000):
      if not self.game_ref.CurrentPlayer.id == C.OPPONENT_ID:  # and self.game_ref.state != python_pb2.Game.COMPLETE:
        break
      assert self.game_ref.CurrentPlayer.id == C.OPPONENT_ID
      self.play_opponent_action()
    else:
      raise TimeoutError

  def __str__(self):
    return f"Player: {self.game_ref.CurrentPlayer.id} - status: {self.game_ref.state} - turns: {self.game_ref.turn}"

  def close(self):
    # Not sure about this.
    # python_pb2_grpc.ServerHandleStub(channel=Sabbertsone.channel).Close(self.game_ref)
    self.logger.warning("Not closing cleanly, restart the server")
