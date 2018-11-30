import config
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import agents.learning.models.ppo
import agents.base_agent
from shared import utils

import baselines.common.vec_env.dummy_vec_env

class PPOAgent(agents.base_agent.Agent):
  def choose(self, observation, possible_actions):
    raise NotImplementedError

  def __init__(self, num_inputs, num_actions, should_flip_board=False,
               model_path="checkpoints/checkpoint.pth.tar", record=True) -> None:
    self.num_inputs = num_inputs
    self.num_actions = num_actions
    # Hyper params:
    self.hidden_size = 256
    self.num_steps = 20
    self.mini_batch_size = 5
    self.ppo_epochs = 4
    self.threshold_reward = 1
    self.model = agents.learning.models.ppo.ActorCritic(num_inputs, num_actions, self.hidden_size)
    self.optimizer = optim.Adam(
      self.model.parameters(),
      lr=config.PPOAgent.lr
    )

  def load_model(self, model_path=None):
    pass

  def ppo_update(self, states, actions, log_probs, returns, advantages, clip_param=0.2):
    ppo_epochs = self.ppo_epochs
    mini_batch_size = self.mini_batch_size

    for _ in range(ppo_epochs):
      for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                       returns, advantages):
        action, dist, value = self.query_model(state, info)

        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(action)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

        actor_loss = - torch.min(surr1, surr2).mean()
        critic_loss = (return_ - value).pow(2).mean()

        loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

  def train(self, load_env):
    envs = [load_env for _ in range(config.PPOAgent.nr_parallel_envs)]
    envs = baselines.common.vec_env.dummy_vec_env.DummyVecEnv(envs)

    states, rewards, dones, infos = envs.reset()

    for frame_idx in range(config.PPOAgent.max_frames):
      actions, log_probs, masks, next_state, rewards, states, values = self.gather_trajectories(envs, states, infos)

      if (frame_idx + 1) % 1000 == 0:
        test_reward = np.mean([test_env() for _ in range(10)])
        self.summary_writer.add_scalar('game_stats/test_rewards', test_reward, step_nr)

      next_state = torch.Tensor(next_state)
      _mu, _std, next_value = self.model(next_state)

      returns = compute_gae(next_value, rewards, masks, values)

      returns = torch.cat(returns).detach()
      log_probs = torch.cat(log_probs).detach()
      values = torch.cat(values).detach()
      states = torch.cat(states)
      actions = torch.cat(actions)
      advantage = returns - values

      self.ppo_update(states, actions, log_probs, returns, advantage)

  def gather_trajectories(self, envs, state, info):
    log_probs = []
    values = []
    states = []
    actions = []
    infos = []
    rewards = []
    masks = []
    entropy = 0

    for traj_step in range(config.PPOAgent.num_steps):

      # TODO: multi env
      assert state.shape[0] == 1
      info, = info

      action, dist, value = self.query_model(state, info)

      next_state, reward, done, info = envs.step(action)

      log_prob = dist.log_prob(action.float())
      entropy += dist.entropy().mean()

      log_probs.append(log_prob)
      values.append(value)
      rewards.append(torch.Tensor(reward).unsqueeze(1))
      masks.append(torch.Tensor(1 - done).unsqueeze(1))

      states.append(state)
      actions.append(action)
      infos.append(info)

      state = next_state
    return actions, log_probs, masks, next_state, rewards, states, values

  def query_model(self, state, info):
    mu, std, value = self.model(state)
    dist = torch.distributions.Normal(mu, std)
    action_distr = dist.sample()
    possible_actions = self.one_hot_actions((info['possible_actions'],))
    possible_actions = torch.Tensor(possible_actions)
    action_prob = F.softmax(action_distr, dim=1)
    action_prob = action_prob * possible_actions
    act_cat = torch.distributions.Categorical(action_prob)
    action = act_cat.sample()
    return action, dist, value

  def one_hot_actions(self, actions):
    return utils.one_hot_actions(actions, self.num_actions)

def test_env(vis=False):
  state = env.reset()
  if vis: env.render()
  done = False
  total_reward = 0
  while not done:
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    dist, _ = model(state)
    next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
    state = next_state
    if vis: env.render()
    total_reward += reward
  return total_reward


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
  values = values + [next_value]
  gae = 0
  returns = []
  for step in reversed(range(len(rewards))):
    delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    gae = delta + gamma * tau * masks[step] * gae
    returns.insert(0, gae + values[step])
  return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
  batch_size = states.size(0)
  for _ in range(batch_size // mini_batch_size):
    rand_ids = np.random.randint(0, batch_size, mini_batch_size)
    yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids,
                                                                                                   :]
