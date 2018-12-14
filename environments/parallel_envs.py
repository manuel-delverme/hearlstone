from baselines.common.vec_env import subproc_vec_env
import numpy as np

from numpy import ndarray
from typing import Tuple
class ParallelEnvs(subproc_vec_env.SubprocVecEnv):
  def reset(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    retr = super(ParallelEnvs, self).reset()
    s, r, t, pa = retr.T
    return np.stack(s, axis=0), np.expand_dims(r, 1), np.expand_dims(t, 1), np.stack(pa, axis=0)

  def step(self, actions: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    s, r, t, pa = super(ParallelEnvs, self).step(actions)
    return np.stack(s, axis=0), np.expand_dims(r, 1), np.expand_dims(t, 1), np.stack(pa, axis=0)
