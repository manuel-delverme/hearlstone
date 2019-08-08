import collections
import pprint
from collections import defaultdict
from typing import Callable

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
from shared.utils import idx_to_one_hot


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


def _make_env(load_env: Callable[[], environments.base_env.BaseEnv]) -> Callable[[], environments.base_env.BaseEnv]:
  def _thunk():
    return load_env()

  return _thunk


def make_vec_envs(load_env: Callable[[int], environments.base_env.BaseEnv], num_processes: int, device: torch.device = hs_config.device
) -> PyTorchCompatibilityWrapper:
  envs = [_make_env(load_env) for _ in range(num_processes)]

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
  return (
    player.hero.atk,
    player.hero.base_health - player.hero.damage,
    player.hero.exhausted,
    player.hero.power.exhausted,
    *pad(player.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4, parse=parse_card),
    *pad(player.board_zone.minions, length=hs_config.Environment.max_cards_in_board * 3, parse=parse_minion),
  )


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

# TODO: hack to maintain compatibility
def parse_card(card, as_idx=False):

  if as_idx is False:
    encoding = get_encoding(card.card_id).flatten()
    return np.concatenate([encoding, [card.atk, card.base_health, card.cost]])
  return np.array([card.card_id, card.atk, card.base_health, card.cost])

def parse_minion(card):
  return np.array([card.atk, card.base_health - card.damage, card.exhausted])


# TODO move me in shared.constants
MINIONS_ONE_HOT = collections.OrderedDict({k: idx_to_one_hot(idx, C.MAX_CARDS) for idx, k in enumerate(C.MINIONS)})
SPELLS_ONE_HOT = collections.OrderedDict({k: idx_to_one_hot(idx, C.MAX_CARDS) for idx, k in enumerate(C.SPELLS)})

def get_encoding(card_id):
  if card_id in C.MINION_IDS:
    return MINIONS_ONE_HOT[card_id].flatten()
  else:
    return SPELLS_ONE_HOT[card_id].flatten()


def pad(x, length, parse):
  _x = []
  for xi in x:
    _x.extend(parse(xi))
  _x.extend((-1,) * (length - len(_x)))
  return _x


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
    *pad(p.hand_zone.entities, length=hs_config.Environment.max_cards_in_hand * 4 * C.MAX_CARDS, parse=parse_card),
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
