import contextlib
import os
from abc import ABC, abstractmethod


class AlreadySteppingError(Exception):
  """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

  def __init__(self):
    msg = 'already running an async step'
    Exception.__init__(self, msg)


class NotSteppingError(Exception):
  """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

  def __init__(self):
    msg = 'not running an async step'
    Exception.__init__(self, msg)


class VecEnv(ABC):
  """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
  closed = False
  viewer = None

  metadata = {
    'render.modes': ['human', 'rgb_array']
  }

  def __init__(self, num_envs, observation_space, action_space):
    self.num_envs = num_envs
    self.observation_space = observation_space
    self.action_space = action_space

  @abstractmethod
  def reset(self):
    """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
    pass

  @abstractmethod
  def step_async(self, actions):
    """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
    pass

  @abstractmethod
  def step_wait(self):
    """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
    pass

  def close_extras(self):
    """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
    pass

  def close(self):
    if self.closed:
      return
    if self.viewer is not None:
      self.viewer.close()
    self.close_extras()
    self.closed = True

  def step(self, actions):
    """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
    self.step_async(actions)
    return self.step_wait()

  @property
  def unwrapped(self):
    if isinstance(self, VecEnvWrapper):
      return self.vectorized_env.unwrapped
    else:
      return self

  def get_viewer(self):
    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.SimpleImageViewer()
    return self.viewer


class VecEnvWrapper(VecEnv):
  """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

  def __init__(self, venv, observation_space=None, action_space=None):
    self.vectorized_env = venv
    VecEnv.__init__(self,
                    num_envs=venv.num_envs,
                    observation_space=observation_space or venv.observation_space,
                    action_space=action_space or venv.action_space)

  def step_async(self, actions):
    self.vectorized_env.step_async(actions)

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def step_wait(self):
    pass

  def close(self):
    return self.vectorized_env.close()

  def render(self, **kwargs):
    return self.vectorized_env.render(**kwargs)

  def get_images(self):
    return self.vectorized_env.get_images()


class VecEnvObservationWrapper(VecEnvWrapper):
  @abstractmethod
  def process(self, obs):
    pass

  def reset(self):
    obs = self.vectorized_env.reset()
    return self.process(obs)

  def step_wait(self):
    obs, rews, dones, infos = self.vectorized_env.step_wait()
    return self.process(obs), rews, dones, infos


class CloudpickleWrapper(object):
  """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

  def __init__(self, x):
    self.x = x

  def __getstate__(self):
    import cloudpickle
    return cloudpickle.dumps(self.x)

  def __setstate__(self, ob):
    import pickle
    self.x = pickle.loads(ob)


@contextlib.contextmanager
def clear_mpi_env_vars():
  """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
  removed_environment = {}
  for k, v in list(os.environ.items()):
    for prefix in ['OMPI_', 'PMI_']:
      if k.startswith(prefix):
        removed_environment[k] = v
        del os.environ[k]
  try:
    yield
  finally:
    os.environ.update(removed_environment)
