from baselines_repo.baselines.common.vec_env.dummy_vec_env import DummyVecEnv as _DummyVecEnv
from baselines_repo.baselines.common.vec_env.vec_normalize import VecNormalize as _VecNormalize
import numpy as np


class DummyVecEnv(_DummyVecEnv):
  def reset(self):
    for e in range(self.num_envs):
      obs, _, _, self.buf_infos[e] = self.envs[e].reset()
      self._save_obs(e, obs)
    return self._obs_from_buf(), _, _, self.buf_infos.copy()


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
    obs, rewards, dones, infos = self.venv.reset()
    filtered_obs = self._obfilt(obs)
    return filtered_obs, rewards, dones, infos

