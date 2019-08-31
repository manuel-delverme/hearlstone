import collections
import functools
import math
import os
import pprint
from collections import defaultdict
from typing import Callable, Optional, List, Text

import numpy as np
import torch

import environments.base_env
import game_utils
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


def _make_env(game_manager: game_utils.GameManager, env_id: str) -> Callable[[], environments.base_env.MultiOpponentEnv]:
  def _thunk():
    return game_manager.instantiate_environment(env_id=env_id)

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


def parse_card(card):
  if card.card_id not in C.CARD_LOOKUP:
    card_draw = 0
    if card.card_id == C.MINIONS.NoviceEngineer:
      card_draw = 1
    elif card.card_id == C.SPELLS.ArcaneIntellect:
      card_draw = 2

    spell_dmg = 0
    if card.card_id == C.SPELLS.Fireball:
      spell_dmg = 6
    elif card.card_id == C.SPELLS.ArcaneExplosion:
      spell_dmg = 1
    elif card.card_id == C.SPELLS.Frostbolt:
      spell_dmg = 3
    elif card.card_id == C.SPELLS.Flamestrike:
      spell_dmg = 4

    card_vec = (
      card.atk,
      card.cost,
      card.base_health,
      spell_dmg,
      card_draw,
      # card.ghostly,
      # card.card_id,
      card.card_id == C.SPELLS.Polymorph,
      card.card_id == C.MINIONS.DoomSayer,
      card.card_id == C.SPELLS.MirrorImage,
      card.card_id == C.SPELLS.TheCoin,
      card.card_id == C.MINIONS.WaterElemental,
      card.card_id == C.MINIONS.GurubashiBerserker,
      card.card_id in (C.SPELLS.Frostbolt, C.SPELLS.FrostNova),  # Freeze
      card.card_id in (C.SPELLS.FrostNova, C.SPELLS.ArcaneExplosion, C.SPELLS.Flamestrike),  # AOE
    )
    C.CARD_LOOKUP[card.card_id] = card_vec
    C.REVERSE_CARD_LOOKUP[card_vec] = card.card_id
  return C.CARD_LOOKUP[card.card_id]


def parse_minion(card):
  return (
    # self.card_id,
    card.atk,
    card.base_health - card.damage,
    # card.num_attacks_this_turn,
    # card.zone_position,
    # card.order_of_play,
    card.exhausted,
    # card.stealth,
    # card.immune,
    # card.charge,
    # card.attackable_by_rush,
    # card.windfury,
    # card.lifesteal,
    card.taunt,
    # card.divine_shield,
    # card.elusive,
    card.frozen,
    card.card_id == C.MINIONS.DoomSayer,
    card.card_id == C.MINIONS.WaterElemental,
    card.card_id == C.MINIONS.GurubashiBerserker,
    card.card_id in (C.MINIONS.OgreMagi, C.MINIONS.KoboldGeomancer, C.MINIONS.Archmage),  # +1 spell damage
    # card.deathrattle,
    # card.silenced
  )


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


def get_empowerment(game):
  return 0.02 * math.log(len(game._options))  # this was 1+log actually


def get_turns_to_letal(player, opponent):
  hero_life, opponent_life = players_life(opponent, player)
  power, value = board_power(player)
  if power > 0:
    reward = max(opponent_life / power, 1.)
  else:
    reward = 0
  return reward


def get_board_adv(game):
  player_power = sum(minion.atk for minion in game.CurrentPlayer.board_zone.minions)
  opponent_power = sum(minion.atk for minion in game.CurrentOpponent.board_zone.minions)

  player_power /= game.CurrentPlayer.base_mana
  opponent_power /= game.CurrentPlayer.base_mana  # both by the player's mana

  reward = math.log(1 + player_power) - math.log(1 + opponent_power)
  reward *= 0.05
  return reward


def get_life_adv(player, opponent):
  hero_life, opponent_life = players_life(opponent, player)
  life_adv = opponent_life - hero_life
  reward = life_adv / hs_config.Environment.max_hero_health_points
  return reward


def get_hand_adv(player, opponent):
  hand_adv = (len(player.hand_zone.entities) - len(opponent.hand_zone.entities))
  draw_adv = (len(player.deck_zone.entities) - len(opponent.deck_zone.entities))  # number of remaining cards
  reward = (hand_adv + draw_adv) / hs_config.Environment.max_cards_in_deck
  return reward


def get_mana_efficiency(player):
  mana_adv = (player.base_mana - player.remaining_mana)
  reward = mana_adv / player.base_mana
  return reward


def players_life(opponent, player):
  opponent_life = opponent.hero.base_health - opponent.hero.damage
  hero_life = player.hero.base_health - player.hero.damage
  return hero_life, opponent_life


# def board_power(player):
#   return power, value


def update_reward(reward, game):
  reward_type = hs_config.Environment.reward_type

  player = game.CurrentPlayer
  opponent = game.CurrentOpponent

  if reward_type == C.RewardType.default:
    extra_reward = 0.
  elif reward_type == C.RewardType.mana_efficency:
    extra_reward = get_mana_efficiency(player)
  elif reward_type == C.RewardType.hand_adv:
    extra_reward = get_hand_adv(player, opponent)
  elif reward_type == C.RewardType.life_adv:
    extra_reward = get_life_adv(player, opponent)
  elif reward_type == C.RewardType.board_adv:
    extra_reward = get_board_adv(player, opponent)
  elif reward_type == C.RewardType.time_left:
    extra_reward = get_turns_to_letal(player, opponent)
  elif reward_type == C.RewardType.empowerment:
    extra_reward = get_empowerment(game)
  else:
    raise NameError("Misspecified reward type")
  return reward + extra_reward


def shape_reward(f):
  @functools.wraps(f)
  def wrapped(self, *f_args, **f_kwargs):
    s, r, d, info = f(self, *f_args, **f_kwargs)
    r = update_reward(r, self.game_snapshot)
    return s, r, d, info

  return wrapped
