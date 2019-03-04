import copy
import os
import pickle
import subprocess
import time
from functools import partial
# TODO use the shmem version
from multiprocessing.pool import Pool

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


class DQNAgent(agents.base_agent.Agent):
  def choose(self, observation, possible_actions):
    raise NotImplementedError

  def __init__(self, num_inputs, num_actions, model_path="checkpoints/evo_checkpoint.pth.tar", record=True,
               population_size=config.ES.population_size, threadcount=config.ES.nr_threads,
               sigma=0.2, learning_rate=0.01, decay=1.0, sigma_decay=1.0, render_test=False, save_path=None
               ) -> None:
    self.nr_threads = threadcount
    self.minibatch = None
    self.num_actions = num_actions
    self.num_inputs = num_inputs
    self.batch_size = config.DQNAgent.batch_size
    self.model_path = model_path
    self.population_size = population_size
    self.sigma = sigma
    self.learning_rate = learning_rate
    self.decay = decay
    self.sigma_decay = sigma_decay

    self.model = nn.Sequential(
      nn.Linear(num_inputs, 100),
      nn.ReLU(True),
      nn.Linear(100, num_actions),
      nn.Softmax()
    )
    experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = 'runs/{}'.format(experiment_name)

    if record:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir, flush_secs=5)
      print("Experiment name:", experiment_name)
      cmd = "find {} -name '*.py' | grep -v venv | tar -cvf {}/code.tar --files-from -".format(os.getcwd(), log_dir)
      subprocess.check_output(cmd, shell=True)
    else:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir='/tmp/trash/', flush_secs=99999)

    self.reward_function = partial(self.fitness, model=self.model)
    try:
      with open(self.model_path, 'rb') as fin:
        self.weights = pickle.load(fin)
    except FileNotFoundError:
      self.weights = list(self.model.parameters())

    self.pool = Pool(threadcount)
    self.render_test = render_test
    self.save_path = save_path

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

    for iter_nr in tqdm.tqdm(range(game_steps)):
      population = []
      for _ in range(self.population_size):
        x = []
        for param in self.weights:
          x.append(np.random.randn(*param.data.size()))
        population.append(x)

      rewards = []
      if config.ES.nr_threads == 1:
        for pop in population:
          weights = self.jitter_weights(copy.deepcopy(self.weights), population=pop)
          reward = self.reward_function(weights)
          rewards.append(reward)
      else:
        ws = [self.jitter_weights(copy.deepcopy(self.weights), population=pop) for pop in population]
        args = zip(ws, envs)
        rewards = self.pool.map(
          self.reward_function,
          args,
        )
      rewards = np.array(rewards)
      self.summary_writer.add_scalar('rewards/mean', rewards.mean(), iter_nr)
      self.summary_writer.add_scalar('rewards/max', rewards.max(), iter_nr)
      self.summary_writer.add_scalar('rewards/std', rewards.std(), iter_nr)
      self.summary_writer.add_histogram('rewards/distr', rewards, iter_nr)

      if np.std(rewards) != 0:
        # normalized_rewards = rewards
        normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
      for index, param in enumerate(self.weights):
          A = np.array([p[index] for p in population])
          rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T).float()
          param.data = param.data + self.learning_rate / (self.population_size * self.sigma) * rewards_pop

          self.learning_rate *= self.decay
          self.sigma *= self.sigma_decay

    final_weights = self.weights
    with open(self.model_path, 'wb') as fout:
      pickle.dump(final_weights, fout)

    reward = self.reward_function((final_weights, make_env), nr_games=1000)
    print(f"Reward from final weights: {reward}")

    """
    observation, reward, terminal, possible_actions = envs.reset()
    progress_bar = tqdm.tqdm(total=game_steps)

    for step_nr, epsilon, beta in iteration_params:
      progress_bar.update(config.DQNAgent.nr_parallel_envs)
      action = self.act(observation, possible_actions, epsilon, step_nr=step_nr)
      next_observation, reward, done, possible_actions = envs.step(action)

      self.learn_from_experience(observation, action, reward, next_observation, done, possible_actions, step_nr, beta)

      observation = next_observation

      self.summary_writer.add_scalar('dqn/epsilon', epsilon, step_nr)
      self.summary_writer.add_scalar('dqn/beta', beta, step_nr)

      for env_idx, (reward_e, done_e) in enumerate(zip(reward, done)):
        if done_e:
          game_value = (reward_e + 1) / 2
          # self.summary_writer.add_scalar('game_stats/end_turn', envs.simulation.game.turn, step_nr)
          self.summary_writer.add_scalar('game_stats/game_value', int(game_value), step_nr * len(done) + env_idx)
          assert reward_e in (-1.0, 0.0, 1.0)
        else:
          assert reward_e < 1

      if step_nr % target_update_every == 0:
        shared.sync_target(self.q_network, self.q_network_target)
      if step_nr % checkpoint_every == 0 and step_nr > 0:
        torch.save(self.q_network.state_dict(), self.model_path)

      # if (step_nr % int(game_steps / 100)) == 0:
      #   self.q_network.value_nosiy_fc3.reset_noise()
      #   self.q_network.value_nosiy_fc3.reset_parameters()
    """

  def __del__(self):
    self.summary_writer.close()
