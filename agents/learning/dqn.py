import math
import random
import collections

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import agents.base_agent

from agents.learning import shared
import tqdm

# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
Variable = lambda *args, **kwargs: autograd.Variable(
  *args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class DQN(nn.Module):
  def __init__(self, num_inputs, num_actions):
    super(DQN, self).__init__()
    self.num_inputs = num_inputs
    self.num_actions = num_actions

    self.layers = nn.Sequential(
      nn.Linear(self.num_inputs + self.num_actions, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),

      # 1 is the Q(s, a) value
      nn.Linear(128, 1),
      nn.Tanh()
    )

  def forward(self, x):
    return self.layers(x)

  def act(self, state, possible_actions, epsilon):
    if random.random() > epsilon:
      network_inputs = []
      for possible_action in possible_actions:
        network_input = np.append(state, possible_action)
        network_inputs.append(network_input)

      network_inputs = np.array(network_inputs)
      network_input = Variable(torch.FloatTensor(network_inputs).unsqueeze(0),
                               volatile=True)
      q_values = self.forward(network_input).cpu().data.numpy()

      best_action = np.argmax(q_values)
      action = possible_actions[best_action]
    else:
      action, = random.sample(possible_actions, 1)
    return action


class DQNAgent(agents.base_agent.Agent):
  def __init__(self,
    num_inputs,
    num_actions,
    gamma,
    model_path="checkpoint.pth.tar",
  ):
    # model = DQN(env.observation_space.shape[0], env.action_space.n)
    self.model = DQN(num_inputs, num_actions)
    # if USE_CUDA:
    #  self.model = self.model.cuda()
    self.optimizer = optim.Adam(self.model.parameters())
    self.replay_buffer = shared.ReplayBuffer(10000)
    self.gamma = gamma
    self.model_path = model_path

  def load_model(self, model_path=None):
    if not model_path:
      model_path = self.model_path
    self.model.load_state_dict(torch.load(model_path))

  def compute_td_loss(self, batch_size):
    state, action, reward, next_state, done, next_actions = self.replay_buffer.sample(
      batch_size)

    state = np.concatenate((state, action), axis=1)
    state = Variable(torch.FloatTensor(np.float32(state)))
    # action = Variable(torch.LongTensor(action))

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

    # q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    q_values = q_values.squeeze()
    next_q_values = next_q_values.squeeze()

    expected_q_values = reward + self.gamma * next_q_values * (1 - done)

    loss = torch.mean((q_values - expected_q_values) ** 2)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss

  def train(
    self, env,
    num_frames,
    eval_every,
    batch_size=32,
    opponent=None,
  ):
    losses = []
    all_rewards = []
    episode_reward = 0
    render_run = False
    scoreboard = {
      'won': 0,
      'lost': 0,
      'draw': 0
    }
    action_stats = collections.defaultdict(int)

    observation, reward, terminal, info = env.reset()
    for frame_idx in tqdm.tqdm(range(1, num_frames + 1)):
      epsilon = shared.epsilon_by_frame(frame_idx, epsilon_decay=num_frames/6)
      possible_actions = info['possible_actions']
      action = self.model.act(observation, possible_actions, epsilon)

      next_state, reward, done, info = env.step(action)
      action_stats[action] += 1

      self.replay_buffer.push(observation, action, reward, next_state, done,
                              info['possible_actions'])

      observation = next_state
      episode_reward += reward

      if done:
        game_value = env.game_value()
        if game_value == 1:
          scoreboard['won'] += 1
        elif game_value == -1:
          scoreboard['lost'] += 1
        elif game_value == 0:
          scoreboard['draw'] += 1

        observation, reward, terminal, info = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

      if len(self.replay_buffer) > batch_size:
        loss = self.compute_td_loss(batch_size)
        losses.append(loss.data[0])

      if frame_idx % eval_every == 0:
        win_ratio = float(scoreboard['won']) / sum(scoreboard.values())
        shared.plot(frame_idx, all_rewards, losses, win_ratio, action_stats, epsilon)

        # for param in self.model.parameters():
        #   print(param.data)

        losses.clear()
        action_stats.clear()
        scoreboard = {
          'won': 0,
          'lost': 0,
          'draw': 0
        }
    torch.save(self.model.state_dict(), self.model_path)

  def choose(self, observation, info):
    possible_actions = info['possible_actions']
    action = self.model.act(observation, possible_actions, 0)
    return action
