import os
from collections import defaultdict
from typing import Callable, Text

import baselines
import numpy as np
import torch
from baselines.common import monitor
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines_repo.baselines.common.vec_env.dummy_vec_env import DummyVecEnv as _DummyVecEnv
from baselines_repo.baselines.common.vec_env.vec_normalize import VecNormalize as _VecNormalize

import environments.base_env
import hs_config
import specs


class DummyVecEnv(_DummyVecEnv):
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
    filtered_obs = self._obfilt(obs)
    return filtered_obs, rewards, dones, infos


class PyTorchCompatibilityWrapper(VecEnvWrapper):
  def __init__(self, venv: baselines.common.vec_env.VecEnv, device: torch.device):
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
    new_infos['possible_actions'] = torch.zeros(
      size=(self.num_envs, self.action_space.n), dtype=torch.float, device=self.device)

    for idx, info in enumerate(infos):
      for k, v in info.items():
        if k not in ('possible_actions',):
          new_infos[k].append(v)
      new_infos['possible_actions'][idx] = torch.from_numpy(info['possible_actions']).float().to(self.device)

    return obs, rewards, dones, new_infos


def _make_env(
  load_env: Callable[[int], environments.base_env.BaseEnv], seed: int, rank: int, log_dir: Text,
  allow_early_resets: bool) -> Callable[[], environments.base_env.BaseEnv]:
  def _thunk():
    env = load_env(seed=seed + rank)
    if log_dir is not None:
      env = monitor.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
    return env

  return _thunk


def make_vec_envs(
  load_env: Callable[[int], environments.base_env.BaseEnv], seed: int, num_processes: int, gamma: float, log_dir: Text,
  device: torch.device, allow_early_resets: bool) -> PyTorchCompatibilityWrapper:
  envs = [_make_env(load_env, seed, process_num, log_dir, allow_early_resets) for process_num in range(num_processes)]

  if len(envs) == 1 or hs_config.VanillaHS.debug:
    vectorized_envs = DummyVecEnv(envs)
  else:
    vectorized_envs = SubprocVecEnv(envs)

  normalized_envs = VecNormalize(vectorized_envs, gamma=gamma)
  pytorch_envs = PyTorchCompatibilityWrapper(normalized_envs, device)
  return pytorch_envs
