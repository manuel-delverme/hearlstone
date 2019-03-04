from typing import Callable

import hs_config
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import agents.learning.models.ppo
import agents.base_agent
from shared import utils

import baselines.common.vec_env.dummy_vec_env

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import agents.learning.a2c_ppo_acktr.algo.ppo
from agents.learning.a2c_ppo_acktr.envs import make_vec_envs
from agents.learning.a2c_ppo_acktr.model import Policy
from agents.learning.a2c_ppo_acktr.storage import RolloutStorage
from agents.learning.a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule


class PPOAgent(agents.base_agent.Agent):
  def choose(self, observation, possible_actions):
    raise NotImplementedError

  def __init__(self, num_inputs, action_space, log_dir: str, should_flip_board=False,
    model_path="checkpoints/checkpoint.pth.tar", record=True, ) -> None:
    self.log_dir = log_dir
    self._setup_logs(log_dir)

    self.num_inputs = num_inputs
    self.num_actions = action_space.n

    actor_critic_network = Policy(num_inputs, action_space, base_kwargs={'recurrent': False})
    actor_critic_network.to(hs_config.device)

    self.agent = agents.learning.a2c_ppo_acktr.algo.ppo.PPO(
      actor_critic_network,
      hs_config.PPOAgent.clip_epsilon,
      hs_config.PPOAgent.ppo_epoch,
      hs_config.PPOAgent.num_mini_batches,
      hs_config.PPOAgent.value_loss_coeff,
      hs_config.PPOAgent.entropy_coeff,
      lr=hs_config.PPOAgent.adam_lr,
      max_grad_norm=hs_config.PPOAgent.max_grad_norm
    )

  def _setup_logs(self, log_dir):
    try:
      os.makedirs(log_dir)
    except OSError:
      # raise ValueError('log dir exists')
      files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
      for f in files:
        os.remove(f)
    eval_log_dir = log_dir + "_eval"
    try:
      os.makedirs(eval_log_dir)
    except OSError:
      # raise ValueError('log dir exists')
      files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
      for f in files:
        os.remove(f)

  #   envs = [load_env for _ in range(config.PPOAgent.num_processes)]
  #   envs = baselines.common.vec_env.dummy_vec_env.DummyVecEnv(envs)

  #   states, rewards, dones, infos = envs.reset()

  #   for frame_idx in range(config.PPOAgent.max_frames):
  #     actions, log_probs, masks, next_state, rewards, states, values = self.gather_trajectories(envs, states, infos)

  #     if (frame_idx + 1) % 1000 == 0:
  #       test_reward = np.mean([test_env() for _ in range(10)])
  #       self.summary_writer.add_scalar('game_stats/test_rewards', test_reward, step_nr)

  #     next_state = torch.Tensor(next_state)
  #     _mu, _std, next_value = self.model(next_state)

  #     returns = compute_gae(next_value, rewards, masks, values)

  #     returns = torch.cat(returns).detach()
  #     log_probs = torch.cat(log_probs).detach()
  #     values = torch.cat(values).detach()
  #     states = torch.cat(states)
  #     actions = torch.cat(actions)
  #     advantage = returns - values

  #     self.ppo_update(states, actions, log_probs, returns, advantage)

  def train(self, load_env: Callable, seed: int, num_processes: int = hs_config.PPOAgent.num_processes):

    envs = make_vec_envs(load_env, seed, num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                         hs_config.PPOAgent.add_timestep, hs_config.device,
                         allow_early_resets=False)

    assert envs.observation_space.shape == self.num_inputs
    assert envs.action_space.n == self.num_actions

    rollouts = RolloutStorage(
      hs_config.PPOAgent.num_steps,
      hs_config.PPOAgent.num_processes,
      envs.observation_space.shape,
      envs.action_space,
      recurrent_hidden_state_size=0,
    )

    obs, _, _, info = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(hs_config.device)

    episode_rewards = deque(maxlen=10)

    start = time.time()

    for ppo_update_num in range(hs_config.PPOAgent.num_updates):
      if hs_config.PPOAgent.use_linear_lr_decay:
        # decrease learning rate linearly
        update_linear_schedule(self.agent.optimizer, ppo_update_num, hs_config.PPOAgent.num_updates, hs_config.PPOAgent.lr)

      if hs_config.PPOAgent.use_linear_clip_decay:
        self.agent.clip_param = hs_config.PPOAgent.clip_param * (1 - ppo_update_num / float(hs_config.PPOAgent.num_updates))

      for step in range(hs_config.PPOAgent.num_steps):
        # sample actions
        with torch.no_grad():
          value, action, action_log_prob, recurrent_hidden_states = self.agent.actor_critic.act(
            rollouts.obs[step],
            rollouts.recurrent_hidden_states[step],
            rollouts.masks[step],
          )

        # obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        for info in infos:
          if 'episode' in info.keys():
            episode_rewards.append(info['episode']['r'])

        # if done then clean the history of observations.
        masks = torch.floattensor([[0.0] if done_ else [1.0]
                                   for done_ in done])
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

    with torch.no_grad():
      next_value = self.agent.actor_critic.get_value(rollouts.obs[-1],
                                                     rollouts.recurrent_hidden_states[-1],
                                                     rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, hs_config.PPOAgent.use_gae, hs_config.PPOAgent.gamma, hs_config.PPOAgent.tau)

    value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
    rollouts.after_update()

    # save for every interval-th episode or for the last epoch
    if (
      ppo_update_num % hs_config.PPOAgent.save_interval == 0 or ppo_update_num == hs_config.PPOAgent.num_updates - 1) and hs_config.PPOAgent.save_dir != "":
      try:
        os.makedirs(hs_config.PPOAgent.save_dir)
      except OSError:
        pass

      # a really ugly way to save a model to cpu
      save_model = self.agent.actor_critic
      if hs_config.use_gpu:
        save_model = copy.deepcopy(self.agent.actor_critic).cpu()

      save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None)]
      torch.save(save_model, os.path.join(hs_config.PPOAgent.save_dir, str(envs[0].game_mode) + ".pt"))

    total_num_steps = (ppo_update_num + 1) * hs_config.PPOAgent.num_processes * hs_config.PPOAgent.num_steps

    if ppo_update_num % hs_config.log_interval == 0 and len(episode_rewards) > 1:
      end = time.time()
      print(
        "updates {}, num timesteps {}, fps {} \n last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
        "min/max reward {:.1f}/{:.1f}\n".
          format(ppo_update_num, total_num_steps,
                 int(total_num_steps / (end - start)),
                 len(episode_rewards),
                 np.mean(episode_rewards),
                 np.median(episode_rewards),
                 np.min(episode_rewards),
                 np.max(episode_rewards), dist_entropy,
                 value_loss, action_loss))

    if hs_config.PPOAgent.eval_interval and len(episode_rewards) > 1 and ppo_update_num % hs_config.PPOAgent.eval_interval == 0:
      eval_envs = make_vec_envs(load_env, seed, num_processes, gamma, self.log_dir, add_timestep, hs_config.device,
                                allow_early_resets=True)

      vec_norm = get_vec_normalize(eval_envs)
      if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

      eval_episode_rewards = []

      obs = eval_envs.reset()
      eval_recurrent_hidden_states = torch.zeros(hs_config.PPOAgent.num_processes,
                                                 self.actor.actor_critic.recurrent_hidden_state_size,
                                                 device=hs_config.device)
      eval_masks = torch.zeros(hs_config.PPOAgent.num_processes, 1, device=hs_config.device)

      while len(eval_episode_rewards) < 10:
        with torch.no_grad():
          _, action, _, eval_recurrent_hidden_states = self.actor.actor_critic.act(
            obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor([[0.0] if done_ else [1.0]
                                   for done_ in done],
                                  dtype=torch.float32,
                                  device=hs_config.device)

        for info in infos:
          if 'episode' in info.keys():
            eval_episode_rewards.append(info['episode']['r'])

      eval_envs.close()

      print(" evaluation using {} episodes: mean reward {:.5f}\n".
            format(len(eval_episode_rewards),
                   np.mean(eval_episode_rewards)))

    # if args.vis and ppo_update_num % args.vis_interval == 0:
    #   try:
    #     # sometimes monitor doesn't properly flush the outputs
    #     win = visdom_plot(viz, win, args.log_dir, args.env_name,
    #                       args.algo, args.num_env_steps)
    #   except ioerror:
    #     pass

# class PPOAgent(agents.base_agent.Agent):
#   def choose(self, observation, possible_actions):
#     raise NotImplementedError
#
#   def __init__(self, num_inputs, num_actions, should_flip_board=False,
#     model_path="checkpoints/checkpoint.pth.tar", record=True) -> None:
#     self.num_inputs = num_inputs
#     self.num_actions = num_actions
#     # Hyper params:
#     self.hidden_size = 256
#     self.num_steps = 20
#     self.mini_batch_size = 5
#     self.ppo_epochs = 4
#     self.threshold_reward = 1
#     self.model = agents.learning.models.ppo.ActorCritic(num_inputs, num_actions, self.hidden_size)
#     self.optimizer = optim.Adam(
#       self.model.parameters(),
#       lr=config.PPOAgent.lr
#     )
#
#   def load_model(self, model_path=None):
#     pass
#
#   def ppo_update(self, states, actions, log_probs, returns, advantages, clip_param=0.2):
#     ppo_epochs = self.ppo_epochs
#     mini_batch_size = self.mini_batch_size
#
#     for _ in range(ppo_epochs):
#       for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
#                                                                        returns, advantages):
#         action, dist, value = self.query_model(state, info)
#
#         entropy = dist.entropy().mean()
#         new_log_probs = dist.log_prob(action)
#
#         ratio = (new_log_probs - old_log_probs).exp()
#         surr1 = ratio * advantage
#         surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
#
#         actor_loss = - torch.min(surr1, surr2).mean()
#         critic_loss = (return_ - value).pow(2).mean()
#
#         loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#   def gather_trajectories(self, envs, state, info):
#     log_probs = []
#     values = []
#     states = []
#     actions = []
#     infos = []
#     rewards = []
#     masks = []
#     entropy = 0
#
#     for traj_step in range(config.PPOAgent.num_steps):
#       # TODO: multi env
#       assert state.shape[0] == 1
#       info, = info
#
#       action, dist, value = self.query_model(state, info)
#
#       next_state, reward, done, info = envs.step(action)
#
#       log_prob = dist.log_prob(action.float())
#       entropy += dist.entropy().mean()
#
#       log_probs.append(log_prob)
#       values.append(value)
#       rewards.append(torch.Tensor(reward).unsqueeze(1))
#       masks.append(torch.Tensor(1 - done).unsqueeze(1))
#
#       states.append(state)
#       actions.append(action)
#       infos.append(info)
#
#       state = next_state
#     return actions, log_probs, masks, next_state, rewards, states, values
#
#   def query_model(self, state, info):
#     mu, std, value = self.model(state)
#     dist = torch.distributions.Normal(mu, std)
#     action_distr = dist.sample()
#     possible_actions = self.one_hot_actions((info['possible_actions'],))
#     possible_actions = torch.Tensor(possible_actions)
#     action_prob = F.softmax(action_distr, dim=1)
#     action_prob = action_prob * possible_actions
#     act_cat = torch.distributions.Categorical(action_prob)
#     action = act_cat.sample()
#     return action, dist, value
#
#   def one_hot_actions(self, actions):
#     return utils.one_hot_actions(actions, self.num_actions)
#
#
# def test_env(vis=False):
#   state = env.reset()
#   if vis: env.render()
#   done = False
#   total_reward = 0
#   while not done:
#     state = torch.FloatTensor(state).unsqueeze(0).to(device)
#     dist, _ = model(state)
#     next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
#     state = next_state
#     if vis: env.render()
#     total_reward += reward
#   return total_reward
#
#
# def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
#   values = values + [next_value]
#   gae = 0
#   returns = []
#   for step in reversed(range(len(rewards))):
#     delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
#     gae = delta + gamma * tau * masks[step] * gae
#     returns.insert(0, gae + values[step])
#   return returns
#
#
# def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
#   batch_size = states.size(0)
#   for _ in range(batch_size // mini_batch_size):
#     rand_ids = np.random.randint(0, batch_size, mini_batch_size)
#     yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids,

