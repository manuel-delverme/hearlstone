import collections
import os
import functools
import pprint
from collections import defaultdict
from typing import Callable, Optional, List, Text

import numpy as np
import torch

import environments.base_env
import hs_config
import specs
from baselines_repo.baselines.common import vec_env
from baselines_repo.baselines.common.vec_env.dummy_vec_env import DummyVecEnv as _DummyVecEnv
from baselines_repo.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines_repo.baselines.common.vec_env.vec_env import VecEnvWrapper
from shared import constants as C


class DummyVecEnv(_DummyVecEnv):
  @property
  def remotes(self):
    return self.envs

  def reset(self):
    for e in range(self.num_envs):
      obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].reset()
      self._save_obs(e, obs)

    return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()


class PyTorchCompatibilityWrapper(VecEnvWrapper):
  def __init__(self, venv: vec_env.VecEnv, device: torch.device):
    super(PyTorchCompatibilityWrapper, self).__init__(venv)
    self.device = device

  def reset(self) -> [torch.FloatTensor, np.ndarray, np.ndarray, specs.Info]:
    transition = self.vectorized_env.reset()
    return self._to_pytorch(*transition, reset=True)

  def step_async(self, actions: torch.Tensor):
    assert actions.shape == (self.num_envs, 1)
    actions = actions.cpu().numpy()
    self.vectorized_env.step_async(actions)

  def step_wait(self) -> [torch.FloatTensor, torch.FloatTensor, np.ndarray, specs.Info]:
    transition = self.vectorized_env.step_wait()
    return self._to_pytorch(*transition)

  def _to_pytorch(self, obs, rewards, dones, infos, reset=False):
    assert obs.shape == (self.num_envs,) + self.observation_space.shape
    assert rewards.shape == (self.num_envs,)
    assert dones.shape == (self.num_envs,)
    assert len(infos) == self.num_envs
    assert not reset or np.all(rewards == 0)
    assert not reset or np.all(~dones)
    assert [specs.check_info_spec(info) for info in infos]

    obs = torch.from_numpy(obs).float().to(self.device)
    rewards = torch.from_numpy(rewards).unsqueeze(dim=1).float()
    dones = torch.from_numpy(dones.astype(np.int32)).unsqueeze(dim=1)

    new_infos = defaultdict(list)
    new_infos['possible_actions'] = torch.zeros(size=(self.num_envs, self.action_space.n), dtype=torch.float,
                                                device=self.device)

    for idx, info in enumerate(infos):
      for k, v in info.items():
        if k not in ('possible_actions',):
          new_infos[k].append(v)
      new_infos['possible_actions'][idx] = torch.from_numpy(info['possible_actions']).float().to(self.device)

    return obs, rewards, dones, dict(new_infos)


def _make_env(load_env: Callable[[], environments.base_env.BaseEnv], env_number: int) -> Callable[[], environments.base_env.BaseEnv]:
  def _thunk():
    return load_env(env_number=env_number)

  return _thunk


def make_vec_envs(name: Text, load_env: Callable[[int], environments.base_env.BaseEnv], num_processes: int,
    device: torch.device = hs_config.device
) -> PyTorchCompatibilityWrapper:
  pid = os.getpid()
  envs = [_make_env(load_env, f"{pid}.{name}.{env_number}") for env_number in range(num_processes)]

  if len(envs) == 1 or hs_config.Environment.single_process:
    vectorized_envs = DummyVecEnv(envs)
  else:
    vectorized_envs = SubprocVecEnv(envs)

  pytorch_envs = PyTorchCompatibilityWrapper(vectorized_envs, device)
  return pytorch_envs


class StdOutWrapper:
  text = ""

  def write(self, txt):
    self.text += txt
    self.text = '\n'.join(self.text.split('\n')[-30:])

  def get_text(self, beg, end):
    return '\n'.join(self.text.split('\n')[beg:end])


def episodic_log(func):
  if not hs_config.Environment.ENV_DEBUG:
    return func

  def wrapper(*args, **kwargs):
    self = args[0]
    if not hasattr(self, '__episodic_log_log_call_depth'):
      self.__episodic_log_log_call_depth = 0

    if not hasattr(self, '__episodic_log_log'):
      self.__episodic_log_log = collections.deque(maxlen=2)

    func_name = func.__name__
    new_episode = func_name == 'reset'

    if new_episode:
      self.__episodic_log_log.append([])
      self.__episodic_log_log_call_depth = 0
    extra_info = tuple()  # (inspect.stack()[1:], r)

    pre = tuple(''.join((' ',) * self.__episodic_log_log_call_depth))

    log_row = tuple(pre + (func_name,) + args[1:] + tuple(kwargs.items()) + extra_info)
    self.__episodic_log_log[-1].append(log_row)

    self.__episodic_log_log_call_depth += 1
    try:
      retr = func(*args, **kwargs)
    except environments.base_env.BaseEnv.GameOver as e:
      raise e
    except Exception as e:
      raise e

    self.__episodic_log_log_call_depth -= 1

    pretty_retr = []
    if retr:
      for r in retr:
        if isinstance(r, dict):
          r = dict(r)
          if 'possible_actions' in r:
            r['possible_actions'] = f'PARSED: {np.argwhere(r["possible_actions"]).flatten()}'
          if 'observation' in r:
            r['observation'] = f'PARSED: min:{r["observation"].min()} max:{r["observation"].max()}'
          pr = str(r)
        elif len(str(r)) > 50:
          post = "..."
          pr = str(r)[:50] + post
        else:
          pr = str(r)
        pretty_retr.append(pr)
    else:
      pretty_retr = retr

    pre = tuple(''.join((' ',) * self.__episodic_log_log_call_depth))
    self.__episodic_log_log[-1].append(tuple(pre + (func_name,) + (pretty_retr,) + extra_info))
    return retr

  return wrapper


def print_log(self):
  log = self.__episodic_log_log
  if log is None:
    return

  for episode in log:
    print('=' * 100)
    for event in episode:
      print(pprint.pformat(event))


def dump_log(self):
  log = self.__episodic_log_log
  if log is None:
    return

  with open('/tmp/episodic_log.txt', 'w') as fout:
    for episode in log:
      fout.write('=' * 100)
      fout.write('\n')
      for event in episode:
        fout.write(pprint.pformat(event, width=4000))
        fout.write('\n')


def parse_player(player):
  raise NotImplementedError
  return (
    player.hero.atk,
    player.hero.base_health - player.hero.damage,
    player.hero.exhausted,
    player.hero.power.exhausted,
    *pad(player.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
    *pad(player.board_zone.minions, length=hs_config.Environment.max_cards_in_board * 3, parse=parse_minion),
  )


def parse_card(card):
  return (card.atk, card.base_health, card.cost) + C.INACTIVE_CARDS_ONE_HOT[card.card_id]


def parse_minion(card):
  return (card.atk, card.base_health - card.damage, card.exhausted, *C.ACTIVE_CARDS_ONE_HOT[card.card_id])


def pad(x: List, length: int, parse: Optional[Callable]):
  _x = []
  for xi in x:
    _x.extend(parse(xi))
  _x.extend((-1,) * (length - len(_x)))
  return _x


def parse_deck(entities):
  entities = sorted(entities, key=lambda x: x.card_id)
  deck = np.ones(shape=(hs_config.Environment.max_cards_in_deck * C.INACTIVE_CARD_ENCODING_SIZE,)) * -1
  old_id = None
  forbiddenf = {}
  forbiddent = {}
  for e in entities:
    card_encoding_pos = C.DECK_ID_TO_POSITION[e.card_id]
    card_encoding_pos += card_encoding_pos == old_id
    assert old_id != card_encoding_pos
    _from, _to = card_encoding_pos * C.INACTIVE_CARD_ENCODING_SIZE, (card_encoding_pos + 1) * C.INACTIVE_CARD_ENCODING_SIZE

    assert _from not in forbiddenf
    assert _to not in forbiddent

    c = parse_card(e)
    assert C.Card(*c)
    assert all(deck[_from: _to] == -1)
    deck[_from: _to] = c
    # print(deck[_from: _to])
    # print(_from, _to, card_encoding_pos)

    old_id = card_encoding_pos
  return deck


def parse_game(game):
  o = game.CurrentOpponent
  p = game.CurrentPlayer

  deck = parse_deck(p.deck_zone.entities)
  assert len(deck) == 390

  p_hand = pad(p.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * C.INACTIVE_CARD_ENCODING_SIZE, parse=parse_card)
  assert len(p_hand) == 130
  p_board = pad(p.board_zone.minions, length=hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE, parse=parse_minion)
  assert len(p_board) == 84 == hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE
  o_board = pad(o.board_zone.minions, length=hs_config.Environment.max_cards_in_board * C.ACTIVE_CARD_ENCODING_SIZE, parse=parse_minion)
  assert len(o_board) == 84

  retr = np.array((
    # player
    p.remaining_mana,
    p.hero.atk,
    p.hero.base_health - p.hero.damage,
    p.hero.exhausted,
    p.hero.power.exhausted,
    *p_hand,
    *p_board,
    *deck,

    # opponent
    o.remaining_mana,
    o.hero.atk,
    o.hero.base_health - o.hero.damage,
    o.hero.exhausted,
    o.hero.power.exhausted,
    # *pad(o.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
    *o_board,
  ), dtype=np.int32)
  assert retr.shape[0] == C.STATE_SPACE
  return retr


def game_stats(game):
  player = game.CurrentPlayer
  opponent = game.CurrentOpponent

  mana_adv = get_mana_efficiency(player)
  hand_adv = get_hand_adv(player, opponent)
  life_adv = get_life_adv(player, opponent)
  n_remaining_turns = get_turns_to_letal(player, opponent)
  board_adv = get_board_adv(player, opponent)

  return C.GameStatistics(mana_adv, hand_adv, life_adv, n_remaining_turns, board_adv)


def get_extra_reward(game, reward_type=None):
  if reward_type == C.RewardType.default:
    return 0.

  player = game.CurrentPlayer
  opponent = game.CurrentOpponent

  if reward_type == C.RewardType.mana_adv:
    reward = get_mana_efficiency(player)
  elif reward_type == C.RewardType.hand_adv:
    reward = get_hand_adv(player, opponent)
  elif reward_type == C.RewardType.life_adv:
    reward = get_life_adv(player, opponent)
  elif reward_type == C.RewardType.board_adv:
    reward = get_board_adv(player, opponent)
  elif reward_type == C.RewardType.time_left:
    reward = get_turns_to_letal(player, opponent)
  else:
    raise NameError("Misspecified reward type")

  return reward


def get_turns_to_letal(player, opponent):
  hero_life, opponent_life = players_life(opponent, player)
  power, value = board_power(player)
  if power > 0:
    reward = max(opponent_life / power, 1.)
  else:
    reward = 0
  return reward


def get_board_adv(player, opponent):
  power, value = board_power(player)
  defense = sum([minion.base_health - minion.damage for minion in opponent.board_zone.minions])
  reward = (value - defense) / max(1,max(value, defense))
  return reward


def get_life_adv(player, opponent):
  hero_life, opponent_life = players_life(opponent, player)
  life_adv = opponent_life - hero_life
  reward = life_adv / hs_config.Environment.max_life
  return reward


def get_hand_adv(player, opponent):
  hand_adv = (len(player.hand_zone.entities) - len(opponent.hand_zone.entities))
  draw_adv = (len(player.deck_zone.entities) - len(opponent.deck_zone.entities))  # number of remaining cards
  reward = (hand_adv + draw_adv) / hs_config.Environment.max_deck_size
  return reward


def get_mana_efficiency(player):
  mana_adv = (player.base_mana - player.remaining_mana)
  reward = mana_adv / player.base_mana
  return reward


def players_life(opponent, player):
  opponent_life = opponent.hero.base_health - opponent.hero.damage
  hero_life = player.hero.base_health - player.hero.damage
  return hero_life, opponent_life


def board_power(player):
  power = [(minion.atk, minion.base_health - minion.damage) for minion in player.board_zone.minions]
  if len(power):
    power, value = np.sum(power, axis=0)
  else:
    power = 0
    value = 0
  return power, value


def shape_reward(f, _lambda=hs_config.Environment.get_reward_shape):
  @functools.wraps(f)
  def wrapped(self, *f_args, **f_kwargs):
    s, r, d, info = f(self, *f_args, **f_kwargs)
    return s, _lambda(r, self.game_snapshot), d, info

  return wrapped
