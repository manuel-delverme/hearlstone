import tensorboardX
import environments.parallel_envs
# TODO use the shmem version
import copy
import numpy as np

import torch
from shared import utils
import agents.base_agent
import random
from typing import Tuple, List

import agents.learning.replay_buffers
from agents.learning import shared
from agents.learning.models import dqn
import tqdm
import hs_config
import os
import time
import torch.nn
import subprocess


class DQNAgent(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions, should_flip_board=False,
               model_path="checkpoints/checkpoint.pth.tar", record=True,
               opponent=None,
               ) -> None:
    self.IMPOSSIBLE_ACTION_PENALTY = -1e5
    self.minibatch = None
    self.num_actions = num_actions
    self.num_inputs = num_inputs
    self.gamma = hs_config.DQNAgent.gamma
    self.batch_size = hs_config.DQNAgent.batch_size
    self.warmup_steps = hs_config.DQNAgent.warmup_steps
    self.model_path = model_path

    self.q_network = dqn.DQN(num_inputs, num_actions)
    self.q_network.build_network()

    self.q_network_target = copy.deepcopy(self.q_network)
    self.q_network_target.build_network()

    optimizer = hs_config.DQNAgent.optimizer

    self.optimizer = optimizer(
      self.q_network.parameters(),
      lr=hs_config.DQNAgent.lr,
      weight_decay=hs_config.DQNAgent.l2_decay,
    )
    self.replay_buffer = agents.learning.replay_buffers.PrioritizedBufferOpenAI(
      hs_config.DQNAgent.buffer_size, num_inputs, num_actions
    )
    self.loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
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
    actions = torch.Tensor(actions).long()
    weights = torch.Tensor(weights)
    rewards = torch.Tensor(rewards)

    if self.minibatch == 0:
      assert (states.shape[0] == self.batch_size), "Invalid shape: " + str(states.shape)
      # assert actions.shape == torch.Size([self.batch_size, self.num_actions]), "Invalid shape: " + str(actions.shape)
      assert actions.shape == torch.Size([self.batch_size, 1]), "Invalid shape: " + str(actions.shape)
      assert rewards.shape == torch.Size([self.batch_size, 1]), "Invalid shape: " + str(rewards.shape)
      assert (next_states.shape == states.shape), "Invalid shape: " + str(next_states.shape)
      assert (dones.shape == rewards.shape), "Invalid shape: " + str(dones.shape)
      # assert (next_possible_actionss.shape == actions.shape), "Invalid shape: " + str(next_possible_actionss.shape)
    self.minibatch += 1

    proj_dist = self.projection_distribution(next_states, rewards, not_done_mask, next_possible_actionss)
    dist = self.q_network(states)
    # next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    actions = actions.unsqueeze(1).expand(self.batch_size, 1, self.q_network.num_atoms)
    dist = dist.gather(1, actions).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    proj_dist = torch.tensor(proj_dist, requires_grad=True)
    loss = -(proj_dist * dist.log()).sum(1)

    # # Compute max a' Q(s', a') over all possible actions using target network
    # next_q_values, max_q_action_idxs = self.get_max_q_values(next_states, next_possible_actionss)

    # target_q_values = rewards + not_done_mask * (self.gamma * next_q_values)

    # # Get Q-value of action taken
    # all_q_values = self.q_network(states)
    # q_values = torch.sum(all_q_values * actions, 1, keepdim=True)

    # loss = self.loss(q_values, target_q_values).squeeze()
    # loss = loss * weights

    assert loss.shape == (self.batch_size,), loss.shape

    # TODO: re-enable
    # priorities = loss * weights + 1e-5
    loss_value = loss.mean()

    self.optimizer.zero_grad()
    loss_value.backward()
    # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), config.DQNAgent.gradient_clip)
    self.optimizer.step()

    self.q_network.reset_noise()
    self.q_network_target.reset_noise()

    # self.replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())

    if self.minibatch % hs_config.DQNAgent.target_update:
      shared.sync_target(self.q_network, self.q_network_target)

    return loss_value

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

  def train(self, make_env, game_steps=None, checkpoint_every=10000,
            target_update_every=hs_config.DQNAgent.target_update, ):
    envs = environments.parallel_envs.ParallelEnvs([make_env] * hs_config.DQNAgent.nr_parallel_envs)

    self.minibatch = None
    if game_steps is None:
      game_steps = hs_config.DQNAgent.training_steps

    epsilon_schedule = shared.epsilon_schedule(
      offset=hs_config.DQNAgent.warmup_steps,
      epsilon_decay=hs_config.DQNAgent.epsilon_decay
    )
    beta_schedule = shared.epsilon_schedule(
      offset=hs_config.DQNAgent.warmup_steps,
      epsilon_decay=hs_config.DQNAgent.beta_decay,
    )
    iteration_params = zip(range(game_steps), epsilon_schedule, beta_schedule)

    self.minibatch = 0
    observation, reward, terminal, possible_actions = envs.reset()
    progress_bar = tqdm.tqdm(total=game_steps)

    for step_nr, epsilon, beta in iteration_params:
      progress_bar.update(hs_config.DQNAgent.nr_parallel_envs)
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

  def choose(self, observation, info):
    board_center = observation.shape[1] // 2
    if self.should_flip_board:
      observation = np.concatenate(observation[board_center:], observation[:board_center], axis=1)

    action = self.q_network.act(observation, info['possible_actions'], 0.0)
    return action

  def learn_from_experience(self, state, action, reward, next_state, done, next_actions, step_nr, beta):
    for i in range(hs_config.DQNAgent.nr_parallel_envs):
      self.replay_buffer.push(state[i], action[i], reward[i], next_state[i], done[i], next_actions[i])

    if len(self.replay_buffer) > max(self.batch_size, self.warmup_steps):
      state, action, reward, next_state, done, next_actions, indices, weights = self.replay_buffer.sample(
        self.batch_size, beta)

      for _ in range(hs_config.DQNAgent.nr_epochs):
        loss = self.train_step(state, action, reward, next_state, done, next_actions, indices, weights)
      self.summary_writer.add_scalar('dqn/loss', loss, step_nr)

  def act(self, state: np.array, possible_actions: np.ndarray, epsilon: float, step_nr: int = None):
    if self.minibatch == 0 or self.minibatch is None:
      assert isinstance(state, np.ndarray)
      assert len(possible_actions.shape) == 2
      # assert all(isinstance(a, int) for a in possible_actions)
      assert isinstance(epsilon, float)

    dist = self.q_network(state).data.cpu()
    dist = dist * torch.linspace(self.q_network.Vmin, self.q_network.Vmax, self.q_network.num_atoms)
    q_values = dist.sum(2)
    q_values, = q_values.numpy()

    # q_values = self.get_q_values(self.q_network, state, possible_actions)
    # q_values, best_action = self.get_max_q_values(state, possible_actions)
    if step_nr is not None:
      self.summary_writer.add_scalar('dqn/minq', min(q_values), step_nr)
      self.summary_writer.add_scalar('dqn/maxq', max(q_values), step_nr)

    possible_actions, = possible_actions
    q_values += -1e5 * (1 - possible_actions)
    action = np.argmax(q_values).reshape(-1, 1)
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
    raise Exception
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

  def projection_distribution(self, next_state, rewards, not_dones, possible_actions):
    Vmin, Vmax, num_atoms = self.q_network.Vmin, self.q_network.Vmax, self.q_network.num_atoms
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)

    next_dist = self.q_network_target(next_state).data.cpu() * support
    next_action = next_dist.sum(2)
    next_action += torch.tensor(-1e5 * (1 - possible_actions))
    next_action = next_action.max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.expand_as(next_dist)
    not_dones = not_dones.expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + not_dones * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (self.batch_size - 1) * num_atoms, self.batch_size).long().unsqueeze(1).expand(
      self.batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist

  def one_hot_actions(self, actions):
    return utils.one_hot_actions(actions, self.num_actions)
