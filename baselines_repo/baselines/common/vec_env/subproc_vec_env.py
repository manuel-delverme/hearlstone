import multiprocessing as mp

import numpy as np

import specs
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


def _flatten_obs(obs):
  assert isinstance(obs, (list, tuple))
  assert len(obs) > 0

  if isinstance(obs[0], dict):
    keys = obs[0].keys()
    return {k: np.stack([o[k] for o in obs]) for k in keys}
  else:
    return np.stack(obs)


def worker(remote, parent_remote, env_fn_wrapper):
  parent_remote.close()
  env = env_fn_wrapper.x()
  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        ob, reward, done, info = env.step(data)
        assert specs.check_info_spec(info)
        remote.send((ob, reward, done, info))
      elif cmd == 'reset':
        ob, reward, done, info = env.reset()
        remote.send((ob, reward, done, info))
      elif cmd == 'render':
        remote.send(env.render(data))
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'get_spaces_spec':
        remote.send((env.observation_space, env.action_space, env.spec))
      elif cmd == 'set_opponents':
        env.set_opponents(**data)
      else:
        raise NotImplementedError
  except KeyboardInterrupt:
    print('SubprocVecEnv worker: got KeyboardInterrupt')
  finally:
    env.close()


class SubprocVecEnv(VecEnv):
  """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

  def __init__(self, env_fns, spaces=None, context='spawn'):
    """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
    self.waiting = False
    self.closed = False
    nenvs = len(env_fns)
    ctx = mp.get_context(context)
    self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(nenvs)])
    self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
               for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
    for p in self.ps:
      p.daemon = True  # if the main process crashes, we should not cause things to hang
      with clear_mpi_env_vars():
        p.start()
    for remote in self.work_remotes:
      remote.close()

    self.remotes[0].send(('get_spaces_spec', None))
    observation_space, action_space, self.spec = self.remotes[0].recv()
    self.viewer = None
    VecEnv.__init__(self, len(env_fns), observation_space, action_space)

  def set_opponents(self, opponents, opponent_dist):
    assert self._assert_not_closed()

    for remote in self.remotes:
      remote.send(('set_opponents', {'opponents': opponents, 'opponent_dist':opponent_dist}))

  def step_async(self, actions):
    assert self._assert_not_closed()
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def step_wait(self):
    assert self._assert_not_closed()
    results = self.gather_transition()
    self.waiting = False
    return results

  def reset(self):
    assert self._assert_not_closed()
    for remote in self.remotes:
      remote.send(('reset', None))
    return self.gather_transition()

  def gather_transition(self):
    results = [remote.recv() for remote in self.remotes]
    obs, rews, dones, infos = zip(*results)
    return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

  def close_extras(self):
    self.closed = True
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()

  def render(self, mode='text'):
    assert self._assert_not_closed()
    for pipe in self.remotes:
      pipe.send(('render', mode))
    imgs = [pipe.recv() for pipe in self.remotes]
    return imgs

  def _assert_not_closed(self):
    assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
    return True

  def __del__(self):
    if not self.closed:
      self.close()
