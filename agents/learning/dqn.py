import tensorboardX
import numpy as np

import torch
import torch.optim as optim
import agents.base_agent

from agents.learning import shared
from agents.learning.models import dqn
from agents.learning.models import dueling_dqn
import tqdm

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
import torch.autograd as autograd


class DQNAgent(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions, gamma,
    should_flip_board=False,
    model_path="checkpoints/checkpoint.pth.tar", ):

    self.model = dqn.DQN(num_inputs, num_actions)
    # self.model = dueling_dqn.DuelingDQN(num_inputs, num_actions)
    self.model.build_network()
    if USE_CUDA:
      self.model = self.model.cuda()
    self.optimizer = optim.Adam(
      self.model.parameters(),
      lr=1e-4,
    )
    self.replay_buffer = shared.ReplayBuffer(10000)
    self.gamma = gamma
    self.should_flip_board = should_flip_board
    self.model_path = model_path
    self.batch_size = 32
    self.summary_writer = tensorboardX.SummaryWriter()

  def load_model(self, model_path=None):
    if model_path is None:
      model_path = self.model_path
    self.model.load_state_dict(torch.load(model_path))
    print('loaded', model_path)

  def compute_td_loss(self, batch_size, step_nr):
    state, action, reward, next_state, done, next_actions = self.replay_buffer.sample(batch_size)

    state = np.concatenate((state, action), axis=1)
    state = Variable(torch.FloatTensor(np.float32(state)))

    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = self.model(state)
    # TODO: fixme, this is just wrong, the assumption is that 1 action then pass
    # it holds for trading game, but nothing else
    next_actions = np.ones_like(action) * -1

    next_state = np.concatenate((next_state, next_actions), axis=1)
    next_state = Variable(torch.FloatTensor(np.float32(next_state)),
                          volatile=True)
    next_q_values = self.model(next_state)

    q_values = q_values.squeeze()
    next_q_values = next_q_values.squeeze()

    expected_q_values = reward + self.gamma * next_q_values * (1 - done)

    loss = torch.mean((q_values - expected_q_values) ** 2)

    self.optimizer.zero_grad()
    loss.backward()
    for n, p in filter(lambda np: np[1].grad is not None, self.model.named_parameters()):
        # print(n, p.grad.data.min(), p.grad.data.max())
        self.summary_writer.add_histogram('grad.' + n, p.grad.data.cpu().numpy(), global_step=step_nr)
        self.summary_writer.add_histogram(n, p.data.cpu().numpy(), global_step=step_nr)

    self.optimizer.step()
    return loss

  def train(self, env, game_steps, checkpoint_every=10000):
    observation, reward, terminal, info = env.reset()
    epsilon_schedule = shared.epsilon_schedule(epsilon_decay=game_steps / 6)

    for step_nr, epsilon in tqdm.tqdm(zip(range(game_steps), epsilon_schedule), total=game_steps):

      self.summary_writer.add_scalar('dqn/epsilon', epsilon, step_nr)
      action = self.model.act(observation, info['possible_actions'], epsilon, step_nr=step_nr)
      next_observation, reward, done, info = env.step(action)
      self.learn_from_experience(observation, action, reward, next_observation, done, info, step_nr)
      self.summary_writer.add_scalar('game_stats/diff_hp', env.simulation.player.hero.health - env.simulation.opponent.hero.health, step_nr)

      observation = next_observation
      if done:
        game_value = env.game_value()
        self.summary_writer.add_scalar('game_stats/end_turn', env.simulation.game.turn, step_nr)
        self.summary_writer.add_scalar('game_stats/game_value', game_value, step_nr)

        assert reward in (-1.0, 0.0, 1.0)
        observation, reward, terminal, info = env.reset()
      else:
        assert abs(reward) < 1
      if step_nr % checkpoint_every == 0:
        torch.save(self.model.state_dict(), self.model_path)

  def choose(self, observation, info):
    possible_actions = info['possible_actions']
    board_size = observation.shape[1]
    board_center = board_size // 2
    if self.should_flip_board:
      observation = np.concatenate(observation[board_center:],
                                   observation[board_center:], axis=1)
    action = self.model.act(observation, possible_actions, 0.0)
    return action

  def learn_from_experience(self, observation, action, reward, next_state, done,
    info, step_nr):
    self.replay_buffer.push(observation, action, reward, next_state, done, info)
    if len(self.replay_buffer) > self.batch_size:
      loss = self.compute_td_loss(self.batch_size, step_nr)
      self.summary_writer.add_scalar('dqn/loss', loss, step_nr)

  def __del__(self):
    self.summary_writer.close()
