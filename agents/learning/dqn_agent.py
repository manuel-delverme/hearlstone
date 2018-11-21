import tensorboardX
import copy
import numpy as np

import torch
import torch.optim as optim
import agents.base_agent
import random
from typing import Tuple, List

import agents.learning.replay_buffers
from agents.learning import shared
from agents.learning.models import dqn
# from agents.learning.models import double_q_learning
import tqdm
import config
from torch.autograd import Variable

USE_CUDA = False


class DQNAgent(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions, should_flip_board=False,
               model_path="checkpoints/checkpoint.pth.tar") -> None:
    self.use_double_q = config.DQNAgent.use_double_q
    self.use_target = config.DQNAgent.use_target
    assert not (self.use_double_q and self.use_target)

    self.num_actions = num_actions
    self.num_inputs = num_inputs
    self.gamma = config.DQNAgent.gamma
    self.batch_size = config.DQNAgent.batch_size
    self.model_path = model_path

    self.q_network = dqn.DQN(num_inputs, num_actions, USE_CUDA)
    self.q_network.build_network()

    if self.use_target or self.use_double_q:
      self.q_network_target = copy.deepcopy(self.q_network)
      self.q_network_target.build_network()

    self.optimizer = optim.Adam(
      self.q_network.parameters(),
      lr=config.DQNAgent.lr,
      weight_decay=config.DQNAgent.l2_decay,
    )
    self.replay_buffer = agents.learning.replay_buffers.PrioritizedBuffer(10000, num_inputs, num_actions)
    self.summary_writer = tensorboardX.SummaryWriter()

  def load_model(self, model_path=None):
    if model_path is None:
      model_path = self.model_path
    self.q_network.load_state_dict(torch.load(model_path))
    print('loaded', model_path)

  def train_step(self, states, actions, rewards, next_states, dones, next_possible_actionss, indices, weights):
    state_action_pairs = np.concatenate((states, actions), axis=1)
    del states, actions

    not_done_mask = dones == 0

    if self.use_target:
      action_selection_network = self.q_network_target
      q_value_network = self.q_network_target
    elif self.use_double_q:
      action_selection_network = self.q_network
      q_value_network = self.q_network_target
    else:
      action_selection_network = self.q_network
      q_value_network = self.q_network

    # TODO: remove loops
    best_future_actions = np.empty(shape=(self.batch_size, self.num_actions))
    for idx, (state, possible_actions) in enumerate(zip(next_states, next_possible_actionss)):
      q_values = self.get_q_values(action_selection_network, state, possible_actions)
      best_future_action_idx = torch.argmax(q_values.detach())
      best_future_actions[idx] = possible_actions[best_future_action_idx]

    next_best_state_action_pairs = np.concatenate((next_states, best_future_actions), axis=1)
    target_future_q_values = q_value_network(next_best_state_action_pairs)
    target_future_q_values = target_future_q_values.detach()

    # what the q_network should have estimated
    discount_tensor = torch.full(target_future_q_values.shape, self.gamma)
    rewards = torch.FloatTensor(rewards)

    expected_q_values = rewards + not_done_mask * (discount_tensor * target_future_q_values)

    # what the q_network estimates
    q_values = self.q_network(state_action_pairs)

    # expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
    weights = torch.FloatTensor(weights)
    loss = (q_values - expected_q_values).pow(2) * weights

    priorities = loss + 1e-5
    loss = loss.mean()

    self.optimizer.zero_grad()
    loss.backward()
    # for n, p in filter(lambda np: np[1].grad is not None, self.model.named_parameters()):
    # self.summary_writer.add_histogram('grad.' + n, p.grad.data.cpu().numpy(), global_step=step_nr)
    # self.summary_writer.add_histogram(n, p.data.cpu().numpy(), global_step=step_nr)
    self.replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())
    self.optimizer.step()

    return loss

  def train(self, env, game_steps, checkpoint_every=10000, target_update_every=100, ):
    observation, reward, terminal, info = env.reset()
    epsilon_schedule = shared.epsilon_schedule(epsilon_decay=game_steps / 6)
    beta_schedule = shared.epsilon_schedule(epsilon_decay=game_steps / 6)

    iteration_params = zip(range(game_steps), epsilon_schedule, beta_schedule)

    for step_nr, epsilon, beta in tqdm.tqdm(iteration_params, total=game_steps):

      action = self.act(observation, info['possible_actions'], epsilon, step_nr=step_nr)

      next_observation, reward, done, info = env.step(action)
      self.learn_from_experience(observation, action, reward, next_observation, done, info['possible_actions'], step_nr, beta)

      observation = next_observation

      self.summary_writer.add_scalar('dqn/epsilon', epsilon, step_nr)

      if done:
        game_value = env.game_value()
        self.summary_writer.add_scalar('game_stats/end_turn', env.simulation.game.turn, step_nr)
        self.summary_writer.add_scalar('game_stats/game_value', (game_value + 1) / 2, step_nr)

        assert reward in (-1.0, 0.0, 1.0)
        observation, reward, terminal, info = env.reset()
      else:
        assert abs(reward) < 1

      if self.use_double_q and step_nr % target_update_every == 0:
        shared.sync_target(self.q_network, self.q_network_target)
      if step_nr % checkpoint_every == 0:
        torch.save(self.q_network.state_dict(), self.model_path)

  def choose(self, observation, info):
    board_center = observation.shape[1] // 2
    if self.should_flip_board:
      observation = np.concatenate(observation[board_center:], observation[:board_center], axis=1)

    action = self.q_network.act(observation, info['possible_actions'], 0.0)
    return action

  def learn_from_experience(self, observation, action, reward, next_state, done, next_actions, step_nr, beta):
    action = np.array(action)
    reward = np.array(reward)
    done = np.array(done)
    self.replay_buffer.push(observation, action, reward, next_state, done, next_actions)
    if len(self.replay_buffer) > self.batch_size:
      state, action, reward, next_state, done, next_actions, indices, weights = self.replay_buffer.sample(
        self.batch_size, beta)
      loss = self.train_step(state, action, reward, next_state, done, next_actions, indices, weights)
      self.summary_writer.add_scalar('dqn/loss', loss, step_nr)

  def act(self, state: np.array, possible_actions: List[Tuple[int, int]], epsilon: float, step_nr: int = None):
    assert isinstance(state, np.ndarray)
    assert isinstance(possible_actions, tuple)
    assert isinstance(possible_actions[0], tuple)
    assert isinstance(possible_actions[0][0], int)
    assert isinstance(epsilon, float)

    if random.random() > epsilon:
      q_values = self.get_q_values(self.q_network, state, possible_actions)
      # if step_nr is not None:
      #   self.summary_writer.add_histogram('q_values', q_values, global_step=step_nr)

      best_action = torch.argmax(q_values)
      action = possible_actions[best_action.detach()]
    else:
      action, = random.sample(possible_actions, 1)
    return action

  @staticmethod
  def get_q_values(q_network, state, possible_actions):
    state_action_pairs = []
    for possible_action in possible_actions:
      state_action_pair = np.append(state, possible_action)
      state_action_pairs.append(state_action_pair)
    q_values = q_network(state_action_pairs)
    return q_values

  def __del__(self):
    self.summary_writer.close()
