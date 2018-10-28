import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import agents.base_agent

from agents.learning import shared

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(
    *args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


class DQNAgent(agents.base_agent.Agent):
    def __init__(self, num_inputs, num_actions, gamma):
        # model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.model = DQN(num_inputs, num_actions)
        if USE_CUDA:
            self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = shared.ReplayBuffer(1000)
        self.gamma = gamma

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, env):
        num_frames = 10000
        batch_size = 32
        gamma = 0.99

        losses = []
        all_rewards = []
        episode_reward = 0

        state = env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = shared.epsilon_by_frame(frame_idx)
            action = self.model.act(state, epsilon)

            next_state, reward, done, _ = env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > batch_size:
                loss = self.compute_td_loss(batch_size)
                losses.append(loss.data[0])

            if frame_idx % 200 == 0:
                shared.plot(frame_idx, all_rewards, losses)

    def choose(self, state, possible_actions):
        pass
