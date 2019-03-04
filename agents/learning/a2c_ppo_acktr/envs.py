import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
# from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from shared.env_utils import SubprocVecEnv
from shared.env_utils import DummyVecEnv
from shared.env_utils import VecNormalize


def make_env(make_env, seed, rank, log_dir, add_timestep, allow_early_resets):
  def _thunk():
    env = make_env()
    env.seed(seed + rank)

    obs_shape = env.observation_space.shape

    if add_timestep and len(
      obs_shape) == 1 and str(env).find('TimeLimit') > -1:
      env = AddTimestep(env)

    if log_dir is not None:
      env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)
    return env

  return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, add_timestep,
  device, allow_early_resets, num_frame_stack=None):
  envs = [make_env(env_name, seed, i, log_dir, add_timestep, allow_early_resets)
          for i in range(num_processes)]

  if len(envs) > 1:
    envs = SubprocVecEnv(envs)
  else:
    envs = DummyVecEnv(envs)

  if len(envs.observation_space.shape) == 1:
    if gamma is None:
      envs = VecNormalize(envs, ret=False)
    else:
      envs = VecNormalize(envs, gamma=gamma)

  envs = VecPyTorch(envs, device)

  if num_frame_stack is not None:
    envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
  elif len(envs.observation_space.shape) == 3:
    envs = VecPyTorchFrameStack(envs, 4, device)

  return envs


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
  def observation(self, observation):
    if self.env._elapsed_steps > 0:
      observation[-2:0] = 0
    return observation


class AddTimestep(gym.ObservationWrapper):
  def __init__(self, env=None):
    super(AddTimestep, self).__init__(env)
    self.observation_space = Box(
      self.observation_space.low[0],
      self.observation_space.high[0],
      [self.observation_space.shape[0] + 1],
      dtype=self.observation_space.dtype)

  def observation(self, observation):
    return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeObs(gym.ObservationWrapper):
  def __init__(self, env=None):
    """
        Transpose observation space (base class)
        """
    super(TransposeObs, self).__init__(env)


class VecPyTorch(VecEnvWrapper):
  def __init__(self, venv, device):
    """Return only every `skip`-th frame"""
    super(VecPyTorch, self).__init__(venv)
    self.device = device
    # TODO: Fix data types

  def reset(self):
    obs, _, _, info = self.venv.reset()
    obs = torch.from_numpy(obs).float().to(self.device)
    return obs, None, None, info

  def step_async(self, actions):
    actions = actions.squeeze(1).cpu().numpy()
    self.venv.step_async(actions)

  def step_wait(self):
    obs, reward, done, info = self.venv.step_wait()
    obs = torch.from_numpy(obs).float().to(self.device)
    reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
    return obs, reward, done, info


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
  def __init__(self, venv, nstack, device=None):
    self.venv = venv
    self.nstack = nstack

    wos = venv.observation_space  # wrapped ob space
    self.shape_dim0 = wos.shape[0]

    low = np.repeat(wos.low, self.nstack, axis=0)
    high = np.repeat(wos.high, self.nstack, axis=0)

    if device is None:
      device = torch.device('cpu')
    self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

    observation_space = gym.spaces.Box(
      low=low, high=high, dtype=venv.observation_space.dtype)
    VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

  def step_wait(self):
    obs, rews, news, infos = self.venv.step_wait()
    self.stacked_obs[:, :-self.shape_dim0] = \
      self.stacked_obs[:, self.shape_dim0:]
    for (i, new) in enumerate(news):
      if new:
        self.stacked_obs[i] = 0
    self.stacked_obs[:, -self.shape_dim0:] = obs
    return self.stacked_obs, rews, news, infos

  def reset(self):
    obs = self.venv.reset()
    if torch.backends.cudnn.deterministic:
      self.stacked_obs = torch.zeros(self.stacked_obs.shape)
    else:
      self.stacked_obs.zero_()
    self.stacked_obs[:, -self.shape_dim0:] = obs
    return self.stacked_obs

  def close(self):
    self.venv.close()
