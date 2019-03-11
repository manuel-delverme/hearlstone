import collections
import copy
import glob
import os
import time
from collections import deque
from typing import Callable

import numpy as np
import torch

import agents.base_agent
import agents.learning.models.ppo
import hs_config
from shared.env_utils import make_vec_envs
from agents.learning.models.randomized_policy import Policy
from agents.learning.shared.storage import RolloutStorage
from shared.utils import get_vec_normalize, update_linear_schedule


class PPOAgent(agents.base_agent.Agent):
  def _choose(self, observation, possible_actions):
    raise NotImplementedError

  def __init__(self, num_inputs, action_space, log_dir: str, should_flip_board=False,
    model_path="checkpoints/checkpoint.pth.tar", record=True, ) -> None:
    self.log = collections.deque(maxlen=2)
    self.log_dir = log_dir
    self._setup_logs(log_dir)

    self.num_inputs = num_inputs
    self.num_actions = action_space.n

    actor_critic_network = Policy(num_inputs, action_space)
    actor_critic_network.to(hs_config.device)

    self.actor_critic = actor_critic_network

    self.clip_param = hs_config.PPOAgent.clip_epsilon
    self.ppo_epoch = hs_config.PPOAgent.ppo_epoch

    self.num_mini_batch = hs_config.PPOAgent.num_mini_batches
    self.value_loss_coef = hs_config.PPOAgent.value_loss_coeff
    self.entropy_coef = hs_config.PPOAgent.entropy_coeff

    self.max_grad_norm = hs_config.PPOAgent.max_grad_norm
    self.use_clipped_value_loss = hs_config.PPOAgent.clip_value_loss

    self.optimizer = torch.optim.Adam(
      self.actor_critic.parameters(),
      lr=hs_config.PPOAgent.adam_lr,
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

  def train(self, load_env: Callable, seed: int, num_processes: int = hs_config.PPOAgent.num_processes):

    envs = make_vec_envs(load_env, seed, num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                         hs_config.device, allow_early_resets=False)

    assert envs.observation_space.shape == self.num_inputs
    assert envs.action_space.n == self.num_actions

    rollouts = RolloutStorage(hs_config.PPOAgent.num_steps, hs_config.PPOAgent.num_processes,
                              envs.observation_space.shape, envs.action_space)

    obs, _, _, info = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.possible_actionss[0].copy_(info[0]['possible_actions'])
    rollouts.to(hs_config.device)

    episode_rewards = deque(maxlen=10)
    possible_actionss = torch.zeros(size=(len(info),) + info[0]['possible_actions'].shape)

    start = time.time()

    for ppo_update_num in range(hs_config.PPOAgent.num_updates):
      if hs_config.PPOAgent.use_linear_lr_decay:
        # decrease learning rate linearly
        update_linear_schedule(self.optimizer, ppo_update_num, hs_config.PPOAgent.num_updates,
                               hs_config.PPOAgent.lr)
      if hs_config.PPOAgent.use_linear_clip_decay:
        self.agent.clip_param = hs_config.PPOAgent.clip_param * (
          1 - ppo_update_num / float(hs_config.PPOAgent.num_updates))

      for step in range(hs_config.PPOAgent.num_steps):
        # sample actions
        with torch.no_grad():
          value, action, action_log_prob = self.actor_critic(rollouts.obs[step], rollouts.possible_actionss[step])

        obs, reward, done, infos = envs.step(action)

        for num, info in enumerate(infos):
          possible_actionss[num] = info['possible_actions']

          if 'episode' in info.keys():
            episode_rewards.append(info['episode']['r'])

        # if done then clean the history of observations.
        dones = [[0.0] if done_ else [1.0] for done_ in done]
        masks = torch.FloatTensor(dones)
        rollouts.insert(obs, action, action_log_prob, value, reward, masks, possible_actionss)

      with torch.no_grad():
        next_value = self.actor_critic.critic(rollouts.obs[-1]).detach()

      rollouts.compute_returns(next_value, hs_config.PPOAgent.use_gae, hs_config.PPOAgent.gamma, hs_config.PPOAgent.tau)

      value_loss, action_loss, dist_entropy = self.update(rollouts)
      rollouts.update_for_new_rollouts()
      total_num_steps = (ppo_update_num + 1) * hs_config.PPOAgent.num_processes * hs_config.PPOAgent.num_steps

      # save for every interval-th episode or for the last epoch
      if (
        ppo_update_num % hs_config.PPOAgent.save_interval == 0 or ppo_update_num == hs_config.PPOAgent.num_updates - 1) and hs_config.PPOAgent.save_dir != "":
        try:
          os.makedirs(hs_config.PPOAgent.save_dir)
        except OSError:
          pass

        # a really ugly way to save a model to cpu
        save_model = self.actor_critic
        if hs_config.use_gpu:
          save_model = copy.deepcopy(self.actor_critic).cpu()

        save_model = [save_model, getattr(get_vec_normalize(envs), 'ob_rms', None)]

        checkpoint_name = "{}-{}.pt".format(hs_config.VanillaHS.get_game_mode().__name__, total_num_steps)
        checkpoint_file = os.path.join(hs_config.PPOAgent.save_dir, checkpoint_name)
        torch.save(save_model, checkpoint_file)

      if ppo_update_num % hs_config.print_every == 0 and len(episode_rewards) > 1:
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

      if hs_config.PPOAgent.eval_interval and len(
        episode_rewards) > 1 and ppo_update_num % hs_config.PPOAgent.eval_interval == 0:

        eval_envs = make_vec_envs(load_env, seed, num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                                  hs_config.device, allow_early_resets=True)
        vec_norm = get_vec_normalize(eval_envs)
        if vec_norm is not None:
          vec_norm.eval()
          vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

        eval_episode_rewards = []

        obs, _, _, infos = eval_envs.reset()
        possible_actionss = torch.zeros(size=(len(infos),) + infos[0]['possible_actions'].shape)

        for num, info in enumerate(infos):
          possible_actionss[num] = info['possible_actions']

        while len(eval_episode_rewards) < 10:
          with torch.no_grad():
            _, action, _ = self.actor_critic(obs, possible_actionss, deterministic=True)
          # obser reward and next obs
          obs, reward, done, infos = eval_envs.step(action)

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

  def enjoy(self, make_env, checkpoint_file):
    env = make_vec_envs(make_env, hs_config.seed, 1, None, None, 'cpu', allow_early_resets=False)
    # We need to use the same statistics for normalization as used in training
    self.actor_critic, ob_rms = torch.load(checkpoint_file)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
      vec_norm.eval()
      vec_norm.ob_rms = ob_rms

    obs, _, _, infos = env.reset()
    possible_actionss = torch.zeros(size=(len(infos),) + infos[0]['possible_actions'].shape)

    for num, info in enumerate(infos):
      possible_actionss[num] = info['possible_actions']

    env.render(info=infos[0])
    while True:
      with torch.no_grad():
        value, action, _, _ = self.actor_critic.forward(obs, None, None, possible_actionss)
      obs, reward, done, infos = env.step(action)
      for num, info in enumerate(infos):
        possible_actionss[num] = torch.from_numpy(info['possible_actions'])

      env.render(info=infos[0])

  def update(self, rollouts):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0

    for e in range(self.ppo_epoch):
      data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

      for sample in data_generator:
        (obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ,
         possible_actions) = sample

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch,
                                                                                    possible_actions)

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
          value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
          value_losses = (values - return_batch).pow(2)
          value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
          value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = 0.5 * (return_batch - values).pow(2).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        value_loss_epoch += value_loss.item()
        action_loss_epoch += action_loss.item()
        dist_entropy_epoch += dist_entropy.item()

    num_updates = self.ppo_epoch * self.num_mini_batch

    value_loss_epoch /= num_updates
    action_loss_epoch /= num_updates
    dist_entropy_epoch /= num_updates

    return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
