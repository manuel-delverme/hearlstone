from baselines.common.vec_env import subproc_vec_env
import numpy as np


class ParallelEnvs(subproc_vec_env.SubprocVecEnv):
  def reset(self):
    retr = super(ParallelEnvs, self).reset()
    s, r, t, pa = (np.stack(e, axis=0) for e in retr.T)
    return s, r, t, pa

  def step(self, actions):
    retr = super(ParallelEnvs, self).step(actions)
    s, r, t, pa = (np.stack(e, axis=0) for e in retr)
    return s, r, t, pa
