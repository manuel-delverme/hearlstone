import numpy as np

import torch
from agents.learning import dqn_agent
import random


class DQNAgent(dqn_agent.DQNAgent):
  def __init__(self, *args, **kwargs):
    self.model = None
    super(DQNAgent, self).__init__(*args, **kwargs)
    assert self.model is not None
    self.target = dqn_agent.DQN(self.num_inputs, self.num_actions)
    self.target.build_network()

  def load_model(self, model_path=None):
    super(DQNAgent, self).load_model(model_path)
    self.update_target(1.0)

  def update_target(self, tau=1.0):
    for t_param, param in zip(self.target.parameters(), self.model.parameters()):
      new_param = tau * param.data + (1.0 - tau) * t_param.data
      t_param.data.copy_(new_param)

  def train_step(self, batch_size, step_nr):
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
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)

    next_q_values = self.model(next_state)

    q_values = q_values.squeeze()
    next_q_values = next_q_values.squeeze()

    expected_q_values = reward + self.gamma * next_q_values * (1 - done)

    loss = torch.mean((q_values - expected_q_values) ** 2)

    self.optimizer.zero_grad()
    loss.backward()
    # for n, p in filter(lambda np: np[1].grad is not None, self.model.named_parameters()):
    #   self.summary_writer.add_histogram('grad.' + n, p.grad.data.cpu().numpy(), global_step=step_nr)
    #   self.summary_writer.add_histogram(n, p.data.cpu().numpy(), global_step=step_nr)

    self.optimizer.step()
    return loss
