import collections
import pprint
import warnings
from collections import defaultdict
from typing import Callable, Text, Optional

import numpy as np
import torch

import environments.base_env
import hs_config
import specs
# from baselines.common import vec_env
from baselines_repo.baselines.common import vec_env
from baselines_repo.baselines.common.vec_env.dummy_vec_env import DummyVecEnv as _DummyVecEnv
from baselines_repo.baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines_repo.baselines.common.vec_env.vec_env import VecEnvWrapper
from baselines_repo.baselines.common.vec_env.vec_normalize import VecNormalize as _VecNormalize
from shared import env_utils


class DummyVecEnv(_DummyVecEnv):
  @property
  def remotes(self):
    return self.envs

  def reset(self):
    for e in range(self.num_envs):
      obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].reset()
      self._save_obs(e, obs)

    return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()


class VecNormalize(_VecNormalize):
  def __init__(self, *args, **kwargs):
    super(VecNormalize, self).__init__(*args, **kwargs)
    self.training = True

  def _obfilt(self, obs):
    if self.ob_rms:
      if self.training:
        self.ob_rms.update(obs)
      obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
      return obs
    else:
      return obs

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def reset(self):
    """
    Reset all environments
    """
    obs, rewards, dones, infos = self.vectorized_env.reset()
    warnings.warn('fixme')
    for oi, ri, info in zip(obs, rewards, infos):
      info['game_statistics'] = (oi, ri)
    filtered_obs = self._obfilt(obs)
    return filtered_obs, rewards, dones, infos


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


def _make_env(
  load_env: Callable[[int], environments.base_env.BaseEnv], rank: int, log_dir: Text,
  allow_early_resets: bool) -> Callable[[], environments.base_env.BaseEnv]:
  def _thunk():
    return load_env(extra_seed=rank)

  return _thunk


def make_vec_envs(
  load_env: Callable[[int], environments.base_env.BaseEnv], num_processes: int, gamma: float, log_dir: Optional[Text],
  device: torch.device, allow_early_resets: bool) -> PyTorchCompatibilityWrapper:
  envs = [_make_env(load_env, process_num, log_dir, allow_early_resets) for process_num in range(num_processes)]

  if len(envs) == 1 or hs_config.Environment.no_subprocess:
    vectorized_envs = DummyVecEnv(envs)
  else:
    vectorized_envs = SubprocVecEnv(envs)

  normalized_envs = VecNormalize(vectorized_envs, gamma=gamma)
  pytorch_envs = PyTorchCompatibilityWrapper(normalized_envs, device)
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
      # env_utils.dump_log(self)
      # print('ENV logs dumped!')
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
