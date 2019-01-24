import copy
import os
import pickle
import subprocess
import time
from functools import partial
# TODO use the shmem version
import multiprocessing

import numpy as np
import tensorboardX
import torch
import torch.nn
import torch.nn as nn
import tqdm

import agents.base_agent
import agents.learning.replay_buffers
import config
from shared.utils import suppress_stdout
import cma


class DQNAgent(agents.base_agent.Agent):
  def choose(self, observation, possible_actions):
    raise NotImplementedError

  def __init__(self, num_inputs, num_actions, model_path="checkpoints/evo_checkpoint.pth.tar", record=True,
               population_size=config.ES.population_size, sigma=0.2) -> None:
    self.model_path = model_path
    self.population_size = population_size

    self.model = nn.Sequential(
      nn.Linear(num_inputs, 100),
      nn.ReLU(True),
      nn.Linear(100, num_actions),
      nn.Softmax()
    )
    self.init_summaries(record)

    self.reward_function = partial(self.fitness, model=self.model)
    with open(self.model_path, 'rb') as fin:
      self.weights = pickle.load(fin)
    self.es = cma.CMAEvolutionStrategy(self.weights, sigma)

  def init_summaries(self, record):
    experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = 'runs/{}'.format(experiment_name)
    if record:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir, flush_secs=5)
      print("Experiment name:", experiment_name)
      cmd = "find {} -name '*.py' | grep -v venv | tar -cvf {}/code.tar --files-from -".format(os.getcwd(), log_dir)
      subprocess.check_output(cmd, shell=True)
    else:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir='/tmp/trash/', flush_secs=99999)

  def load_model(self, model_path=None):
    raise NotImplementedError

  @staticmethod
  def fitness(hax, model, render=False, nr_games=config.ES.nr_games):
    weights, make_env = hax
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
      try:
        param.data.copy_(weights[i])
      except:
        param.data.copy_(weights[i].data)

    with suppress_stdout():
      env = make_env()
    total_reward = 0

    for _ in range(nr_games):
      observation, reward, done, possible_actions = env.reset()

      while not done:
        if render:
          env.render()
          time.sleep(0.05)

        with suppress_stdout():
          prediction = cloned_model(torch.Tensor(observation))
        p = prediction.detach().numpy()
        action = (p - 99999 * (1 - possible_actions)).argmax()
        observation, reward, done, possible_actions = env.step(action)

        total_reward += reward

    env.close()
    return total_reward / nr_games

  def jitter_weights(self, weights, population=[], no_jitter=False):
    new_weights = []
    for i, param in enumerate(weights):
      if no_jitter:
        new_weights.append(param.data)
      else:
        jittered = torch.from_numpy(self.sigma * population[i]).float()
        new_weights.append(param.data + jittered)
    return new_weights

  def train(self, make_env, game_steps=None, checkpoint_every=10000):
    # envs = environments.parallel_envs.ParallelEnvs([make_env] * config.DQNAgent.nr_parallel_envs)
    # TODO: start the envs here
    envs = [make_env for _ in range(self.population_size)]

    if game_steps is None:
      game_steps = config.ES.nr_iters
    with multiprocessing.Pool(config.ES.nr_threads) as pool:
      for iter_nr in tqdm.tqdm(range(game_steps)):
        weights = self.es.ask()
        args = zip(weights, envs)
        rewards = pool.map(self.reward_function, args)
        rewards = np.array(rewards)
        self.es.tell(weights, rewards)
        self.summary_writer.add_scalar('rewards/std', rewards.std(), iter_nr)
        rewards = (rewards + 1) / 2
        self.summary_writer.add_scalar('rewards/mean', rewards.mean(), iter_nr)
        self.summary_writer.add_scalar('rewards/max', rewards.max(), iter_nr)
        self.summary_writer.add_histogram('rewards/distr', rewards, iter_nr)

    final_weights = self.es.best
    with open(self.model_path, 'wb') as fout:
      pickle.dump(final_weights, fout)

    reward = self.reward_function((final_weights, make_env), nr_games=1000)
    print(f"Reward from final weights: {reward}")

  def __del__(self):
    self.summary_writer.close()
