import tensorboardX
import numpy as np

import torch
import torch.optim as optim
import agents.base_agent
import random
from typing import Tuple, List

import agents.learning.replay_buffers
from agents.learning import shared
from agents.learning.models import dqn
import tqdm
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
USE_CUDA = False


def Variable(*args, **kwargs):
  v = autograd.Variable(*args, **kwargs)
  if USE_CUDA:
    v.cuda()
  return v


class DQNAgent(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions, gamma, should_flip_board=False,
               use_target=True, model_path="checkpoints/checkpoint.pth.tar"):

    self.network = dqn.DQN(num_inputs, num_actions)
    self.network.build_network()
    if USE_CUDA:
      self.network = self.network.cuda()
    params = self.network.parameters()

    self.use_target = use_target
    if use_target:
      self.target_network = dqn.DQN(num_inputs, num_actions)
      self.target_network.build_network()

      if USE_CUDA:
        self.target_network = self.target_network.cuda()

    self.optimizer = optim.Adam(params, lr=1e-5, )
    self.replay_buffer = agents.learning.replay_buffers.PrioritizedBuffer(10000, num_inputs, num_actions)
    self.gamma = gamma
    self.should_flip_board = should_flip_board
    self.model_path = model_path
    self.batch_size = 128
    self.summary_writer = tensorboardX.SummaryWriter()

  def load_model(self, model_path=None):
    if model_path is None:
      model_path = self.model_path
    self.network.load_state_dict(torch.load(model_path))
    print('loaded', model_path)

  def compute_td_loss(self, state, action, reward, next_state, done, next_actions, indices, weights):
    state_action = np.concatenate((state, action), axis=1)
    state_action = Variable(torch.FloatTensor(np.float32(state_action)))

    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    weights = Variable(torch.FloatTensor(weights))

    # TODO: fixme, this is just WRONG, the assumption is that 1 action then pass
    # TODO: it holds for trading game, but nothing else, not even the tradingHS
    # TODO: this breaks many things
    next_actions = np.ones_like(action) * -1

    next_state_action = np.concatenate((next_state, next_actions), axis=1)
    next_state_action = Variable(torch.FloatTensor(np.float32(next_state_action)))

    q_values = self.network(state_action)

    if self.use_target:
      # not needded because of the above TODOs
      # next_q_values = self.network(next_state_action)
      next_q_values_target = self.target_network(next_state_action)
      # FIXME: the action should be selected by the network and the future
      # FIXME: value should be selected by the target network
      next_q_values = next_q_values_target
    else:
      next_q_values = self.network(next_state_action)

    expected_q_values = reward + self.gamma * next_q_values * (1 - done)

    loss = (q_values - expected_q_values.detach()).pow(2) * weights
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

      if self.use_target and step_nr % target_update_every == 0:
        shared.sync_target(self.network, self.target_network)
      if step_nr % checkpoint_every == 0:
        torch.save(self.network.state_dict(), self.model_path)

  def choose(self, observation, info):
    board_center = observation.shape[1] // 2
    if self.should_flip_board:
      observation = np.concatenate(observation[board_center:], observation[:board_center], axis=1)

    action = self.network.act(observation, info['possible_actions'], 0.0)
    return action

  def learn_from_experience(self, observation, action, reward, next_state, done, next_actions, step_nr, beta):
    action = np.array(action)
    reward = np.array(reward)
    done = np.array(done)
    self.replay_buffer.push(observation, action, reward, next_state, done, next_actions)
    if len(self.replay_buffer) > self.batch_size:
      state, action, reward, next_state, done, next_actions, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
      loss = self.compute_td_loss(state, action, reward, next_state, done, next_actions, indices, weights)
      self.summary_writer.add_scalar('dqn/loss', loss, step_nr)

  def act(self, state: np.array, possible_actions: List[Tuple[int, int]], epsilon: float, step_nr: int = None):
    assert isinstance(state, np.ndarray)
    assert isinstance(possible_actions, tuple)
    assert isinstance(possible_actions[0], tuple)
    assert isinstance(possible_actions[0][0], int)
    assert isinstance(epsilon, float)

    if random.random() > epsilon:
      network_inputs = []
      for possible_action in possible_actions:
        network_input = np.append(state, possible_action)
        network_inputs.append(network_input)

      network_inputs = np.array(network_inputs)
      network_input = Variable(torch.FloatTensor(network_inputs).unsqueeze(0))
      q_values = self.network.forward(network_input).data.cpu().numpy()

      # if step_nr is not None:
      #   self.summary_writer.add_histogram('q_values', q_values, global_step=step_nr)

      best_action = np.argmax(q_values)
      action = possible_actions[best_action]
    else:
      action, = random.sample(possible_actions, 1)
    return action

  def __del__(self):
    self.summary_writer.close()
