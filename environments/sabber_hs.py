import warnings
from typing import Tuple, Dict, Text, Any

import frozendict
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


def parse_card(card):
  return card.card_id, card.atk, card.base_health, card.cost


def parse_minion(card):
  return card.atk, card.base_health - card.damage, card.exhausted


def pad(x, length, parse):
  _x = []
  for xi in x:
    _x.extend(parse(xi))
  _x.extend((-1,) * (length - len(_x)))
  return _x


def parse_player(player):
  return (
    player.hero.atk,
    player.hero.base_health - player.hero.damage,
    player.hero.exhausted,
    player.hero.power.exhausted,
    *pad(player.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
    *pad(player.board_zone.minions, length=hs_config.Environment.max_cards_in_board * 3, parse=parse_minion),
  )


def parse_game(game):
  o = game.CurrentOpponent
  p = game.CurrentPlayer

  return np.array((
    # player
    p.remaining_mana,
    p.hero.atk,
    p.hero.base_health - p.hero.damage,
    p.hero.exhausted,
    p.hero.power.exhausted,
    *pad(p.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
    *pad(p.board_zone.minions, length=hs_config.Environment.max_cards_in_board * 3, parse=parse_minion),

    # opponent
    o.remaining_mana,
    o.hero.atk,
    o.hero.base_health - o.hero.damage,
    o.hero.exhausted,
    o.hero.power.exhausted,
    # *pad(o.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
    *pad(o.board_zone.minions, length=hs_config.Environment.max_cards_in_board * 3, parse=parse_minion),
  ), dtype=np.int32)


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
  assert len(id_to_action) == C._ACTION_SPACE
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

  return C.GameStatistics(mana_adv, hand_adv, draw_adv, life_adv, n_remaining_turns, minion_adv)
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


class Sabbertsone(environments.base_env.RenderableEnv):
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

    self.action_space = gym.spaces.Discrete(n=C._ACTION_SPACE)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(C._STATE_SPACE,), dtype=np.int)
    self.turn_stats = []
    self._game_matrix = {}
    self.logger.info(f"Env with id {env_number} started.")

  def cards_in_hand(self):
    raise len(self.game_snapshot.CurrentPlayer.hand_zone)

  def game_value(self):
    player = self.game_snapshot.CurrentPlayer
    if player.play_state == sabberstone_protobuf.Controller.WON:  # maybe PlayState
      reward = 1
    elif player.play_state in (sabberstone_protobuf.Controller.LOST, sabberstone_protobuf.Controller.TIED):
      reward = -1
    else:
      reward = 0
    return np.array(reward, dtype=np.float32)

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

  def update_stats(self):
    if self.game_snapshot.CurrentPlayer.id == 1:
      new_stats = game_stats(self.game_snapshot)
      self.turn_stats.append(new_stats)

  @shared.env_utils.episodic_log
  def step(self, action_id: np.ndarray, auto_reset: bool = True):
    assert hasattr(action_id, '__int__')
    action_id = int(action_id)
    stepping_player = self.game_snapshot.CurrentPlayer.id

    self.logger.info(self)

    try:
      with self.logger("parse_options"):
        actions = self.parse_options(self.game_snapshot)

      selected_action = actions[action_id]

      with self.logger("update_stats"):
        self.update_stats()

      assert self.game_snapshot.state != sabberstone_protobuf.Game.COMPLETE
      with self.logger("call_process"):
        self.game_snapshot = self.stub.Process(selected_action)

      if self.game_snapshot.turn > hs_config.Environment.max_turns:
        state, reward, done, info = self.gather_transition(auto_reset=auto_reset)
        done = True
        reward = 0.
        return state, reward, done, info

      if self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID and stepping_player == C.AGENT_ID:  # and self.game_ref.state != python_pb2.Game.COMPLETE:
        self.play_opponent_turn()

    except self.GameOver:
      assert self.game_snapshot.state == sabberstone_protobuf.Game.COMPLETE

      if hs_config.Environment.ENV_DEBUG_METRICS:
        self.logger.error(f"GameOver called from player {self.game_snapshot.CurrentPlayer.id}")

    return self.gather_transition(auto_reset=auto_reset)

  @shared.env_utils.episodic_log
  def reset(self):
    self.logger.info(f"Reset called from player {self.game_snapshot.CurrentPlayer.id}")
    with self.logger("call_reset"):
      self.game_snapshot = self.stub.Reset(self.game_snapshot.id)
    # TODO make me formal
    if np.random.uniform() < 0.2 or self.opponent is None:
      self._sample_opponent()
    # self.turn_stats = []
    # self.episode_steps = 0
    # self.info = None
    if self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID:
      self.play_opponent_turn()
    return self.gather_transition(auto_reset=False)

  @shared.env_utils.episodic_log
  def gather_transition(self, auto_reset: bool) -> Tuple[np.ndarray, np.ndarray, bool, Dict[Text, Any]]:
    assert shared.utils.can_autoreset(auto_reset,
                                      self.game_snapshot) or self.game_snapshot.turn > hs_config.Environment.max_turns

    terminal = self.game_snapshot.state == sabberstone_protobuf.Game.COMPLETE
    assert self.game_snapshot.state in (
      sabberstone_protobuf.Game.INVALID, sabberstone_protobuf.Game.LOADING, sabberstone_protobuf.Game.RUNNING,
      sabberstone_protobuf.Game.COMPLETE,)

    with self.logger("get_value"):
      reward = self.game_value()

    with self.logger("build_state"):
      state = parse_game(self.game_snapshot)

    possible_actions = np.zeros(C._ACTION_SPACE, dtype=np.float32)
    if not terminal:
      actions = self.parse_options(self.game_snapshot)
      possible_actions[list(actions.keys())] = 1

    if terminal:
      if auto_reset:
        if self.game_snapshot.state == sabberstone_protobuf.Game.COMPLETE:
          if self.game_snapshot.CurrentPlayer.id != C.AGENT_ID:
            reward = - reward

        # self.logger.log_stats(game_stats)
        with self.logger("reset_env"):
          state, _, _, _info = self.reset()
        possible_actions = _info['possible_actions']
      else:
        raise self.GameOver

    info = {
      'possible_actions': possible_actions,
    }
    if terminal:
      # TODO maybe make me better
      self.game_matrix(self.current_k, reward)

      counts = np.array([v[1] for v in self._game_matrix.values()])

      _stats = C.GameStatistics(*zip(*self.turn_stats))
      info['game_statistics'] = {
        **{'mean_' + k: v for k, v in zip(C.GameStatistics._fields, np.mean(_stats, axis=1))},
        'outcome': reward,
        'life_adv': self.turn_stats[-1].life_adv,
        'mean_opponent': counts.mean(),
      }

    info = frozendict.frozendict(info)
    assert info['possible_actions'].max() == 1 or terminal
    self.last_info = info  # for rendering render
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
    # self.opponent = self.opponents[-1]

    if self.opponent_obs_rmss is not None:
      self.opponent_obs_rmss = self.opponent_obs_rmss[k]

    self.current_k = k

  def play_opponent_action(self):
    assert self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID
    with self.logger("opponent_step"):
      observation, _, terminal, info = self.gather_transition(auto_reset=False)

    if self.opponent_obs_rms is not None:
      raise NotImplementedError
      observation = (observation - self.opponent_obs_rms.mean) / np.sqrt(self.opponent_obs_rms.var)

    bot_info = dict(info)
    bot_info['original_info'] = {
      "game_snapshot": self.game_snapshot,
      "game_options": self.parse_options(self.game_snapshot),
    }
    action = self.opponent.choose(observation=observation, info=bot_info)
    assert self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID
    return self.step(action, auto_reset=False)

  def play_opponent_turn(self):
    for _ in range(1000):
      if not self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID:  # and self.game_ref.state != python_pb2.Game.COMPLETE:
        break

      self.play_opponent_action()
    else:
      raise TimeoutError

  def __str__(self):
    return f"Player: {self.game_snapshot.CurrentPlayer.id} - status: {self.game_snapshot.state} - turns: {self.game_snapshot.turn}"

  def close(self):
    self.logger.warning("Not closing cleanly, restart the server")
