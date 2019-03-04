import tensorboardX
# import environments.parallel_envs
# TODO use the shmem version
import copy
import numpy as np
# import torch.nn.functional as F

import torch

from environments import vanilla_hs
from shared import utils
import agents.base_agent
import random
from typing import Tuple, List, Callable

import agents.learning.replay_buffers
from agents.learning import shared
from agents.learning.models import dqn
import tqdm
import config
import os
import time
import torch.nn
import subprocess


class DQNAgent(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions, model_path="checkpoints/checkpoint.pth.tar", record=True,
               experiment_name="") -> None:
    self.IMPOSSIBLE_ACTION_PENALTY = -1e5
    self.minibatch_num = None
    self.num_actions = num_actions
    self.num_inputs = num_inputs
    self.gamma = config.DQNAgent.gamma
    self.batch_size = config.DQNAgent.batch_size
    self.warmup_steps = config.DQNAgent.warmup_steps
    self.model_path = model_path

    self.q_network = dqn.DQN(num_inputs, num_actions)
    self.q_network.build_network()

    self.q_network_target = copy.deepcopy(self.q_network)
    self.q_network_target.build_network()

    optimizer = config.DQNAgent.optimizer

    self.optimizer = optimizer(
      self.q_network.parameters(),
      lr=config.DQNAgent.lr,
      weight_decay=config.DQNAgent.l2_decay,
    )
    self.replay_buffer = agents.learning.replay_buffers.PrioritizedBufferOpenAI(
      config.DQNAgent.buffer_size, num_inputs, num_actions
    )
    self.loss = torch.nn.SmoothL1Loss(reduction='none')
    experiment_name += time.strftime("%Y_%m_%d-%H_%M_%S")

    log_dir = 'runs/{}'.format(experiment_name)

    if record:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir)
      print("Experiment name:", experiment_name)
      cmd = "find {} -name '*.py' | grep -v venv | tar -cvf {}/code.tar --files-from -".format(os.getcwd(), log_dir)
      subprocess.check_output(cmd, shell=True)
    else:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir='/tmp/trash/', flush_secs=10)

  def load_model(self, model_path=None):
    if model_path is None:
      model_path = self.model_path
    self.q_network.load_state_dict(torch.load(model_path))
    print('loaded', model_path)

  def train_step(self, states, actions, rewards, next_states, dones, next_possible_actionss, indices, weights):
    # actions = self.one_hot_actions(actions.reshape(-1, 1))
    not_done_mask = torch.Tensor(1 - dones.astype(np.int))
    # next_possible_actionss = self.one_hot_actions(next_possible_actionss)

    states = torch.Tensor(states).detach().requires_grad_(True)
    actions = torch.Tensor(actions)
    weights = torch.Tensor(weights)
    rewards = torch.Tensor(rewards)

    if self.minibatch_num == 0:
      assert (states.shape[0] == self.batch_size), "Invalid shape: " + str(states.shape)
      # assert actions.shape == torch.Size([self.batch_size, self.num_actions]), "Invalid shape: " + str(actions.shape)
      assert actions.shape == torch.Size([self.batch_size, 1]), "Invalid shape: " + str(actions.shape)
      assert rewards.shape == torch.Size([self.batch_size, 1]), "Invalid shape: " + str(rewards.shape)
      assert (next_states.shape == states.shape), "Invalid shape: " + str(next_states.shape)
      assert (dones.shape == rewards.shape), "Invalid shape: " + str(dones.shape)
      # assert (next_possible_actionss.shape == actions.shape), "Invalid shape: " + str(next_possible_actionss.shape)
    self.minibatch_num += 1

    # Compute max a' Q(s', a') over all possible actions using target network
    next_q_values, max_q_action_idxs = self.get_max_q_values(next_states, next_possible_actionss)

    target_q_values = rewards + not_done_mask * (self.gamma * next_q_values)

    # Get Q-value of action taken
    all_q_values = self.q_network(states)
    q_values = torch.sum(all_q_values * actions, 1, keepdim=True)

    loss_dqn = self.loss(q_values, target_q_values).squeeze()
    loss_dqn = loss_dqn * weights

    assert loss_dqn.shape == (self.batch_size,), loss_dqn.shape

    priorities = loss_dqn * weights + 1e-5
    loss = loss_dqn.mean()

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), config.DQNAgent.gradient_clip)
    self.optimizer.step()

    self.replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())

    if self.minibatch_num % config.DQNAgent.target_update:
      shared.sync_target(self.q_network, self.q_network_target)

    return loss, 0, 0

  def render(self, env):
    observation, reward, terminal, possible_actions = env.reset()
    while True:
      q_values = self.get_q_values(self.q_network, observation, possible_actions).detach().numpy()
      best_action = np.argmax(q_values)
      action = possible_actions[best_action]

      print('=' * 100)
      raise NotImplementedError
      # for a, q in zip(info['original_info']['possible_actions'], q_values):
      #   print(a, q)
      print(env.render(info={}))

      observation, reward, done, possible_actions = env.step(action)

      if done:
        print(done, reward)
        observation, reward, terminal, possible_actions = env.reset()

  def train(self, make_env: Callable[[], vanilla_hs.VanillaHS], game_steps=None, checkpoint_every=10000,
            target_update_every=config.DQNAgent.target_update, eval_every=config.DQNAgent.eval_every):
    env = make_env()

    # Gather some edge states.
    almost_lost_states = []
    for _ in range(1):
      env.reset(shuffle_deck=True)
      obs, _, _, _ = env.cheat_hp(player_hp=1, opponent_hp=config.VanillaHS.starting_hp)
      almost_lost_states.append(obs)

    almost_won_states = []
    for _ in range(1):
      env.reset(shuffle_deck=True)
      obs, _, _, _ = env.cheat_hp(player_hp=config.VanillaHS.starting_hp, opponent_hp=1)
      almost_won_states.append(obs)
    env.reset()

    self.minibatch_num = None
    if game_steps is None:
      game_steps = config.DQNAgent.training_steps

    epsilon_schedule = shared.epsilon_schedule(
      offset=config.DQNAgent.warmup_steps,
      epsilon_decay=config.DQNAgent.epsilon_decay
    )
    beta_schedule = shared.epsilon_schedule(
      offset=config.DQNAgent.warmup_steps,
      epsilon_decay=config.DQNAgent.beta_decay,
    )
    iteration_params = zip(range(game_steps), epsilon_schedule, beta_schedule)

    self.minibatch_num = 0
    observation, reward, terminal, possible_actions = env.reset()
    progress_bar = tqdm.tqdm(total=game_steps)
    cum_reward = 0

    for step_nr, epsilon, beta in iteration_params:
      progress_bar.update(config.DQNAgent.nr_parallel_envs)
      action = self.act(observation, possible_actions, epsilon, step_nr=step_nr)
      next_observation, reward, done, possible_actions = env.step(action)
      self.summary_writer.add_scalar('dqn/reward', reward, step_nr)
      cum_reward += reward

      self.learn_from_experience(observation, action, reward, next_observation, done, possible_actions, step_nr, beta)

      observation = next_observation

      if done:
        self.summary_writer.add_scalar('dqn/epsilon', epsilon, step_nr)
        self.summary_writer.add_scalar('dqn/beta', beta, step_nr)
        self.summary_writer.add_scalar('game_stats/end_turn', env.simulation.game.turn, step_nr)
        # self.summary_writer.add_scalar('game_stats/game_value', int(0.5 + reward / 2), step_nr)
        self.summary_writer.add_scalar('game_stats/cum_reward', cum_reward, step_nr)
        cum_reward = 0
        assert reward in (-1.0, 0.0, 1.0)
        observation, reward, terminal, possible_actions = env.reset()
      else:
        if env.game_mode == 'normal':
          assert reward < 1

      if step_nr % eval_every == 0 and step_nr > 0:
        good_values = self.q_network.get_value(almost_won_states)
        bad_values = self.q_network.get_value(almost_lost_states)
        self.summary_writer.add_scalar('debug/almost_won_value', good_values.mean(), step_nr)
        self.summary_writer.add_scalar('debug/almost_lost_value', bad_values.mean(), step_nr)
        self.summary_writer.add_scalar('debug/eval_gap', good_values.mean() - bad_values.mean(), step_nr)

        num_won_games = 0.0
        for _ in range(config.DQNAgent.eval_episodes):
          observation, reward, done, possible_actions = env.reset()
          while not done:
            action = self.act(observation, possible_actions, 0.0, step_nr=None)
            observation, reward, done, possible_actions = env.step(action)
          if env.game_mode == 'normal':
            assert reward in (-1.0, 1.0)
          num_won_games += int(0.5 + reward / 2)

        self.summary_writer.add_scalar('game_stats/win_ratio', num_won_games / config.DQNAgent.eval_episodes, step_nr)

      if step_nr % target_update_every == 0:
        shared.sync_target(self.q_network, self.q_network_target)

      if step_nr % checkpoint_every == 0 and step_nr > 0:
        torch.save(self.q_network.state_dict(), self.model_path)

  def choose(self, observation, info):
    action = self.q_network.act(observation, info['possible_actions'], 0.0)
    return action

  def learn_from_experience(self, observation, action, reward, next_state, done, next_actions, step_nr, beta):
    action = np.array(action)
    reward = np.array(reward)
    done = np.array(done)
    self.replay_buffer.push(observation, action, reward, next_state, done, next_actions)
    if len(self.replay_buffer) > max(self.batch_size, self.warmup_steps):
      state, action, reward, next_state, done, next_actions, indices, weights = self.replay_buffer.sample(
        self.batch_size, beta)

      loss, loss_good, loss_bad = self.train_step(state, action, reward, next_state, done, next_actions, indices,
                                                  weights)
      self.summary_writer.add_scalar('dqn/loss', loss, step_nr)
      self.summary_writer.add_scalar('dqn/loss_good', loss_good, step_nr)
      self.summary_writer.add_scalar('dqn/loss_bad', loss_bad, step_nr)

  def act(self, state: np.array, possible_actions: np.ndarray, epsilon: float, step_nr: int = None) -> int:
    if self.minibatch_num == 0 or self.minibatch_num is None:
      assert isinstance(state, np.ndarray)
      assert len(possible_actions.shape) == 1
      assert all(a.dtype == np.float32 for a in possible_actions)
      assert isinstance(epsilon, float)

    q_values = self.q_network(state).data.cpu().numpy()

    if step_nr is not None:
      self.summary_writer.add_scalar('dqn/minq', min(q_values), step_nr)
      self.summary_writer.add_scalar('dqn/maxq', max(q_values), step_nr)
      self.summary_writer.add_scalar('debug/pass_value', q_values[0], step_nr)

    q_values += -1e5 * (1 - possible_actions)
    action = int(np.argmax(q_values))
    assert possible_actions[action] == 1
    return action

  def __del__(self):
    self.summary_writer.close()

  def get_max_q_values(self, states, possible_actions):
    """
    Used in Q-learning update.
    :param states: Numpy array with shape (batch_size, state_dim). Each row
        contains a representation of a state.
    :param possible_actions: Numpy array with shape (batch_size, action_dim).
        possible_next_actions[i][j] = 1 iff the agent can take action j from
        state i.
    :param double_q_learning: bool to use double q-learning
    """
    q_values = self.q_network(states).detach()
    q_values_target = self.q_network_target(states).detach()
    # Set q-values of impossible actions to a very large negative number.
    inverse_pna = 1 - possible_actions
    impossible_action_penalty = self.IMPOSSIBLE_ACTION_PENALTY * inverse_pna
    # TODO: is it needded?
    q_values += torch.Tensor(impossible_action_penalty)
    # Select max_q action after scoring with online network
    max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
    # Use q_values from target network for max_q action from online q_network
    # to decouble selection & scoring, preventing overestimation of q-values
    q_values = torch.gather(q_values_target, 1, max_indicies)
    return q_values, max_indicies

  def one_hot_actions(self, actions):
    return utils.one_hot_actions(actions, self.num_actions)
