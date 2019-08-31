import warnings

import grpc
import gym
import numpy as np

import environments.base_env
import hs_config
import pysabberstone.python.rpc.python_pb2 as sabberstone_protobuf
import pysabberstone.python.rpc.python_pb2_grpc as sabberstone_grpc
import shared.env_utils
import shared.utils
from shared import constants as C
from shared.env_utils import parse_deck, pad, parse_card, parse_minion


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
    self._deck1 = None
    self._deck2 = None
    self._cards = None

  def NewGame(self, deck1, deck2):
    self._deck1 = deck1
    self._deck2 = deck2
    return _GameRef(self.stub.NewGame(sabberstone_protobuf.DeckStrings(deck1=deck1, deck2=deck2)))

  def Reset(self, game):
    assert isinstance(game, _GameRef)
    return self.NewGame(self._deck1, self._deck2)

  def Process(self, game_id, selected_action):
    return _GameRef(self.stub.Process(selected_action))

  def GetOptions(self, game):
    assert isinstance(game, _GameRef)
    return self.stub.GetOptions(game.game_ref.id).list

  def LoadCards(self):
    self._cards = self.stub.GetCardDictionary(sabberstone_protobuf.Empty()).cards

  def GetCard(self, idx):
    if self._cards is None:
      self.LoadCards()
    return self._cards[idx]


def enumerate_actions():
  pass_action = (C.PlayerTaskType.END_TURN, 0, 0)
  id_to_action = [pass_action]

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

  action_to_id_dict[(C.PlayerTaskType.END_TURN, 1, 2)] = action_to_id_dict[pass_action]
  action_to_id_dict[(C.PlayerTaskType.END_TURN, 2, 1)] = action_to_id_dict[pass_action]

  assert len(id_to_action) == C.ACTION_SPACE
  return action_to_id_dict


class Sabberstone(environments.base_env.RenderableEnv):
  action_to_id = enumerate_actions()
  board_to_board = {C.PlayerTaskType.MINION_ATTACK, C.PlayerTaskType.HERO_ATTACK, C.PlayerTaskType.HERO_POWER}

  def __init__(self, *, address: str = None, seed: int = None, env_number: int = None):
    assert address is not None or env_number is not None
    super().__init__()
    self.gui = None

    if seed is not None:
      warnings.warn("Setting the seed is not implemented")

    self.connect(address, env_number)
    self.game_snapshot = self.stub.NewGame(deck1=C.DECK1, deck2=C.DECK2)

    self.action_space = gym.spaces.Discrete(n=C.ACTION_SPACE)
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(C.STATE_SPACE,), dtype=np.int)
    self.turn_stats = {k: [] for k in C.GameStatistics._fields}  # TODO: do this in game_stats initilaization everywhere
    self._game_matrix = {}

  def connect(self, address, env_number):
    self.channel = grpc.insecure_channel(address)
    self.stub = Stub(sabberstone_grpc.SabberStonePythonStub(self.channel))

  def game_value_for_player(self):
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
      options = self.stub.GetOptions(game)
      possible_options = {}
      # self.source_position,; self.target_position,; self.sub_option,; self.choice
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

  @shared.env_utils.episodic_log
  @shared.env_utils.shape_reward
  @shared.env_utils.episodic_log
  def step(self, action_id: np.ndarray):
    assert_terminal, info, observation, rewards = self.resolve_action(action_id)

    self.turn_stats['empowerment'].append(shared.env_utils.get_empowerment(self.game_snapshot))
    # self.turn_stats['mana_adv'].append(shared.env_utils.get_mana_efficiency(self.game_snapshot))
    # self.turn_stats['hand_adv'].append(shared.env_utils.get_hand_adv(self.game_snapshot))
    # self.turn_stats['life_adv'].append(shared.env_utils.get_life_adv(self.game_snapshot))
    # self.turn_stats['n_remaining_turns'].append(shared.env_utils.get_turns_to_letal(self.game_snapshot))
    # self.turn_stats['board_adv'].append(shared.env_utils.get_board_adv(self.game_snapshot)

    terminal = any(rewards)
    assert not assert_terminal or terminal
    if terminal:
      reward = [r for r in rewards if r != 0.][0]
      info['game_statistics'] = {
        **{f"episode_{k}": sum(v) for k, v in self.turn_stats.items() if v},
        'outcome': reward,
        'opponent_nr': self.current_k,
      }
      for v in self.turn_stats.values():
        v.clear()
    else:
      reward = 0.

    self.last_info = info
    self.last_observation = observation

    return observation, reward, terminal, info

  def resolve_action(self, action_id):
    assert_terminal = False
    rewards = []
    while True:
      self.game_snapshot = self.stub.Process(self.game_snapshot, self.action_int_to_obj(action_id))
      rewards.append(self.game_value_for_player())
      _terminal = self.game_snapshot.state == sabberstone_protobuf.Game.COMPLETE

      if _terminal:
        observation, _, _, info = self.reset()
        assert_terminal = True
      else:
        observation, possible_actions = self.parse_game()
        info = {'possible_actions': possible_actions, }

      if self.game_snapshot.CurrentPlayer.id == C.OPPONENT_ID:
        action_id = self.opponent.choose(
            deterministic=self.current_opponent_is_deterministic,
            observation=observation,
            info={**info, 'original_info': {
              "game_snapshot": self.game_snapshot,
              "game_options": self.parse_options(self.game_snapshot),
            }})
      else:
        break
    return assert_terminal, info, observation, rewards

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
    self.game_snapshot = self.stub.Reset(self.game_snapshot)

    observation, possible_actions = self.parse_game()
    info = {'possible_actions': possible_actions, }
    self.last_info = info
    self.last_observation = observation

    if not check_hand_size(self.game_snapshot.CurrentPlayer.hand_zone):
      raise ValueError(
          f'found {list(self.game_snapshot.CurrentPlayer.hand_zone)} as starting hand, not valid')
    return observation, 0, False, info

  def _sample_opponent(self):
    if self.opponent is not None and hs_config.GameManager.newest_opponent_prob > np.random.uniform():
      return

    p = self.opponent_distribution
    p /= p.sum()

    k = np.random.choice(np.arange(0, len(self.opponents)), p=p)
    self.logger.info(f"Sampled new opponent with id {k} and prob {p[k]}")
    self.opponent = self.opponents[k]
    self.current_k = k

  def __str__(self):
    return f"Player: {self.game_snapshot.CurrentPlayer.id} - status: {self.game_snapshot.state} - turns: {self.game_snapshot.turn}"

  def close(self):
    self.logger.warning("Not closing cleanly, restart the server")

  def __del__(self):
    del self.gui

  def parse_game(self):
    p, o, p_hand, p_board, o_board, deck = self.gather_zones(self.game_snapshot)
    obs = np.array((
      # player
      p.remaining_mana,
      p.hero.atk,
      p.hero.base_health - p.hero.damage,
      p.hero.exhausted,
      p.hero.power.exhausted,
      *p_hand,
      *p_board,
      *deck,
      len(p.board_zone.minions),

      # opponent
      o.remaining_mana,
      o.hero.atk,
      o.hero.base_health - o.hero.damage,
      o.hero.exhausted,
      o.hero.power.exhausted,
      # *pad(o.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
      *o_board,
      len(o.board_zone.minions),
    ), dtype=np.int32)
    pa = self.gather_possible_actions()
    assert obs.shape[0] == C.STATE_SPACE
    assert pa.shape[0] == C.ACTION_SPACE
    return obs, pa

  @staticmethod
  def gather_zones(game_snapshot):
    p = game_snapshot.CurrentPlayer
    o = game_snapshot.CurrentOpponent

    deck = parse_deck(p.deck_zone.entities)
    assert len(deck) == 390
    p_hand = pad(p.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * C.INACTIVE_CARD_ENCODING_SIZE, parse=parse_card)
    assert len(p_hand) == 130
    p_board = pad(p.board_zone.minions, length=hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE, parse=parse_minion)
    assert len(p_board) == hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE
    o_board = pad(o.board_zone.minions, length=hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE, parse=parse_minion)
    assert len(o_board) == hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE

    return p, o, p_hand, p_board, o_board, deck


def check_hand_size(hand_zone):
  if hasattr(hand_zone, 'count'):
    count = hand_zone.count
  else:
    count = len(hand_zone.entities)
  return count in (4, 5)
