import warnings

import grpc
import gym
import numpy as np

import environments.base_env
import hs_config
import sb_env.SabberStone_python_client.python_pb2 as sabberstone_protobuf
import sb_env.SabberStone_python_client.python_pb2_grpc as sabberstone_grpc
import shared.constants as C
import shared.env_utils
import shared.utils
from shared.env_utils import parse_game


class _GameRef:
  def __init__(self, game_ref):
    self.game_ref = game_ref

  @property
  def id(self):
    return self.game_ref.id

  @property
  def turn(self):
    return self.game_ref.turn

  @property
  def state(self):
    return self.game_ref.state

  @property
  def CurrentPlayer(self):
    return self.game_ref.CurrentPlayer

  @property
  def CurrentOpponent(self):
    return self.game_ref.CurrentOpponent


class Stub:
  def __init__(self, stub):
    self.stub = stub

  def NewGame(self, deck1, deck2):
    return _GameRef(self.stub.NewGame(sabberstone_protobuf.DeckStrings(deck1=deck1, deck2=deck2)))

  def Reset(self, game_id):
    return _GameRef(self.stub.Reset(game_id))

  def Process(self, selected_action):
    return _GameRef(self.stub.Process(selected_action))

  def GetOptions(self, game_id):
    return self.stub.GetOptions(game_id).list


def enumerate_actions():
  id_to_action = [(C.PlayerTaskType.END_TURN, 0, 0)]

  # place minions or play spell
  for src_id in range(hs_config.Environment.max_cards_in_hand):
    for target_id in range(hs_config.Environment.max_entities_in_board * 2):
      id_to_action.append((C.PlayerTaskType.PLAY_CARD, C.HandPosition(src_id), C.BoardPosition(target_id)))

  # attack
  for src_id in range(hs_config.Environment.max_entities_in_board):
    for target_id in range(hs_config.Environment.max_entities_in_board,
                           hs_config.Environment.max_entities_in_board * 2):
      id_to_action.append((C.PlayerTaskType.MINION_ATTACK, C.BoardPosition(src_id), C.BoardPosition(target_id)))

  # hero power`
  for target_id in range(hs_config.Environment.max_entities_in_board * 2):
    id_to_action.append((C.PlayerTaskType.HERO_POWER, C.BoardPosition(0), C.BoardPosition(target_id)))

  # hero attack
  for target_id in range(hs_config.Environment.max_entities_in_board, hs_config.Environment.max_entities_in_board * 2):
    id_to_action.append((C.PlayerTaskType.HERO_ATTACK, C.BoardPosition(0), C.BoardPosition(target_id)))

  action_to_id_dict = {v: k for k, v in enumerate(id_to_action)}
  assert len(id_to_action) == C.ACTION_SPACE
  return action_to_id_dict


class Sabberstone(environments.base_env.RenderableEnv):
  DECK1 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="
  DECK2 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="

  action_to_id = enumerate_actions()

  hand_encoding_size = 4  # atk, health, exhaust
  hero_encoding_size = 4  # atk, health, exhaust, hero_power
  minion_encoding_size = 3  # atk, health, exhaust
  board_to_board = {C.PlayerTaskType.MINION_ATTACK, C.PlayerTaskType.HERO_ATTACK, C.PlayerTaskType.HERO_POWER}

  def __init__(self, address: str, seed: int = None, env_number: int = None):
    super().__init__()
    self.gui = None
    self.logger = shared.utils.HSLogger(__class__.__name__, log_to_stdout=hs_config.log_to_stdout)

    if seed is not None:
      warnings.warn("Setting the seed is not implemented")

    if env_number is not None:
      warnings.warn("Setting the seed is not implemented")
      self.extra_seed = env_number

    with self.logger("call_init"):
      self.channel = grpc.insecure_channel(address)
      self.stub = Stub(sabberstone_grpc.SabberStonePythonStub(self.channel))
      self.game_snapshot = self.stub.NewGame(deck1=self.DECK1, deck2=self.DECK2)

    self.action_space = gym.spaces.Discrete(n=C.ACTION_SPACE)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(C.STATE_SPACE,), dtype=np.int)
    self.turn_stats = []
    self.game_matrix = None
    self.logger.info(f"Env with id {env_number} started.")

  def agent_game_vale(self):
    player = self.game_snapshot.CurrentPlayer
    if player.play_state == sabberstone_protobuf.Controller.WON:  # maybe PlayState
      reward = 1
    elif player.play_state in (sabberstone_protobuf.Controller.LOST, sabberstone_protobuf.Controller.TIED):
      reward = -1
    else:
      reward = 0

    if player.id == C.OPPONENT_ID:
      reward = -reward
    return np.float32(reward)

  def parse_options(self, game, return_options=False):
    if not hasattr(game, '_options'):
      options = self.stub.GetOptions(game.id)
      possible_options = {}

      for option in options:
        option_type = option.type
        assert option_type != C.PlayerTaskType.CHOOSE

        if option_type == C.PlayerTaskType.END_TURN:
          action_id = self.GameActions.PASS_TURN
        else:
          source = option.source_position
          target = option.target_position

          assert (option_type not in (C.PlayerTaskType.HERO_ATTACK, C.PlayerTaskType.HERO_POWER)
                  or source == C.BoardPosition.Hero)

          action_hash = (option_type, source, target)
          action_id = self.action_to_id[action_hash]

        assert action_id not in possible_options
        possible_options[action_id] = option

      assert len(possible_options) > 0  # pass should always be there
      game._options = options
      game._possible_options = possible_options

    if return_options:
      return game._possible_options, game._options
    else:
      return game._possible_options

  # def update_stats(self):
  #   if self.game_snapshot.CurrentPlayer.id == 1:
  #     new_stats = game_stats(self.game_snapshot)
  #     self.turn_stats.append(new_stats)

  @shared.env_utils.episodic_log
  def step(self, action_id: np.ndarray):
    rewards = []
    while True:
      self.game_snapshot = self.stub.Process(self.action_int_to_obj(action_id))
      rewards.append(self.agent_game_vale())
      _terminal = self.game_snapshot.state == sabberstone_protobuf.Game.COMPLETE

      if _terminal:
        self.game_snapshot = self.stub.Reset(self.game_snapshot.id)

      observation = parse_game(self.game_snapshot)
      info = {'possible_actions': self.gather_possible_actions(), }

      if self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID:
        action_id = self.opponent.choose(
            observation=observation,
            info={**info, 'original_info': {
              "game_snapshot": self.game_snapshot,
              "game_options": self.parse_options(self.game_snapshot),
            }})
      else:
        break

    terminal = any(rewards)
    if terminal:
      reward = [r for r in rewards if r != 0.][0]
      info['game_statistics'] = self.gather_game_statistics(reward)
      self.game_matrix[self.current_k] += [max(0., -reward), 1]  # prob of p2 winning, number of matches
    else:
      reward = 0.

    observation = parse_game(self.game_snapshot)
    self.last_info = info
    self.last_observation = observation

    return observation, reward, terminal, info

  def gather_possible_actions(self):
    possible_actions = np.zeros(C.ACTION_SPACE, dtype=np.float32)
    possible_actions[list(self.parse_options(self.game_snapshot).keys())] = 1.
    return possible_actions

  def action_int_to_obj(self, action_id):
    action_id = int(action_id)
    actions = self.parse_options(self.game_snapshot)
    selected_action = actions[action_id]
    return selected_action

  @shared.env_utils.episodic_log
  def reset(self):
    self._sample_opponent()
    observation = parse_game(self.game_snapshot)
    possible_actions = np.zeros(C.ACTION_SPACE, dtype=np.float32)
    possible_actions[list(self.parse_options(self.game_snapshot).keys())] = 1
    info = {
      'possible_actions': possible_actions,
    }
    self.last_info = info
    self.last_observation = observation
    return observation, 0, False, info

  def set_opponents(self, opponents):
    super(Sabberstone, self).set_opponents(opponents)
    self.game_matrix = np.zeros(shape=(len(self.opponents), 2))

  def _sample_opponent(self):
    if hs_config.Environment.newest_opponent_prob > np.random.uniform() and self.opponent is not None:
      return
    p = np.ones(shape=len(self.opponents))

    if len(self.opponents) > 1:
      if self.game_matrix[:, 1].sum() > 0:
        # TODO update me with probability of winning
        idxs = self.game_matrix[:, 1] > 0  # only for the one who played
        p[idxs] = 1 / self.game_matrix[:, 1][idxs]
      p /= p.sum()

    k = np.random.choice(np.arange(0, len(self.opponents)), p=p)
    self.logger.info(f"Sampled new opponent with id {k} and prob {p[k]}")
    self.opponent = self.opponents[k]
    self.current_k = k

  def gather_game_statistics(self, reward):
    # counts = np.array([v[1] for v in self._game_matrix.values()])
    # _stats = C.GameStatistics(*zip(*self.turn_stats))
    return {
      # **{'mean_' + k: v for k, v in zip(C.GameStatistics._fields, np.mean(_stats, axis=1))},
      'outcome': reward,
      # 'life_adv': self.turn_stats[-1].life_adv,
      # 'mean_opponent': counts.mean(),
      'opponent_nr': self.current_k,
    }

  def __str__(self):
    return f"Player: {self.game_snapshot.CurrentPlayer.id} - status: {self.game_snapshot.state} - turns: {self.game_snapshot.turn}"

  def close(self):
    self.logger.warning("Not closing cleanly, restart the server")
