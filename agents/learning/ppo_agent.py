import copy
import glob
import itertools
import os
import time
from collections import deque
from typing import Callable

import gym.spaces
import numpy as np
import torch

import agents.base_agent
import agents.learning.models.ppo
import hs_config
import specs
from agents.learning.models.randomized_policy import Policy
from agents.learning.shared.storage import RolloutStorage
from shared.env_utils import make_vec_envs
from shared.utils import get_vec_normalize


class PPOAgent(agents.base_agent.Agent):
  def _choose(self, observation, possible_actions):
    raise NotImplementedError

  def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete, log_dir: str) -> None:
    assert isinstance(observation_space, gym.spaces.Box)
    assert len(observation_space.shape) == 1

    assert isinstance(action_space, gym.spaces.Discrete)
    assert action_space.n > 1, 'the agent can only pass'

    self._setup_logs(log_dir)
    self.log_dir = log_dir

    self.num_inputs = observation_space.shape[0]
    self.num_actions = action_space.n

    actor_critic_network = Policy(self.num_inputs, self.num_actions)
    actor_critic_network.to(hs_config.device)
    self.actor_critic = actor_critic_network

    self.clip_param = hs_config.PPOAgent.clip_epsilon
    assert specs.check_positive_type(self.clip_param, float)

    self.ppo_epoch = hs_config.PPOAgent.ppo_epoch
    assert specs.check_positive_type(self.ppo_epoch, int)

    self.num_mini_batch = hs_config.PPOAgent.num_mini_batches
    assert specs.check_positive_type(self.num_mini_batch, int)

    self.value_loss_coef = hs_config.PPOAgent.value_loss_coeff
    assert specs.check_positive_type(self.value_loss_coef, float)

    self.entropy_coef = hs_config.PPOAgent.entropy_coeff
    assert specs.check_positive_type(self.entropy_coef, float)

    self.max_grad_norm = hs_config.PPOAgent.max_grad_norm
    assert specs.check_positive_type(self.max_grad_norm, float)

    self.use_clipped_value_loss = hs_config.PPOAgent.clip_value_loss
    assert isinstance(self.use_clipped_value_loss, bool)

    self.num_processes = hs_config.PPOAgent.num_processes
    assert specs.check_positive_type(self.num_processes, int)

    self.model_dir = hs_config.PPOAgent.save_dir
    assert isinstance(self.model_dir, str)

    self.save_every = hs_config.PPOAgent.save_interval
    assert specs.check_positive_type(self.save_every, int)
    self.num_updates = hs_config.PPOAgent.num_updates
    assert specs.check_positive_type(self.num_updates, int)
    self.eval_every = hs_config.PPOAgent.eval_interval
    assert specs.check_positive_type(self.eval_every, int)

    self.optimizer = torch.optim.Adam(
      self.actor_critic.parameters(),
      lr=hs_config.PPOAgent.adam_lr,
    )

  @staticmethod
  def _setup_logs(log_dir):
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

  def train(self, load_env: Callable, seed: int):

    envs = make_vec_envs(load_env, seed, self.num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                         hs_config.device, allow_early_resets=False)

    assert envs.observation_space.shape == (self.num_inputs,)
    assert envs.action_space.n == self.num_actions

    rollouts = RolloutStorage(self.num_inputs, self.num_actions)
    obs, _, _, info = envs.reset()
    rollouts.store_first_transition(obs, info['possible_actions'])
    episode_rewards = deque(maxlen=10)

    start = time.time()

    for ppo_update_num in range(hs_config.PPOAgent.num_updates):

      def stop_training(rews, step):
        return step >= hs_config.PPOAgent.num_steps

      self.gather_rollouts(rollouts, episode_rewards, envs, stop_training, False)

      with torch.no_grad():
        next_value = self.actor_critic.critic(rollouts.get_last_observation()).detach()

      rollouts.compute_returns(next_value)

      value_loss, action_loss, dist_entropy = self.update(rollouts)
      rollouts.roll_over_last_transition()
      total_num_steps = (ppo_update_num + 1) * hs_config.PPOAgent.num_processes * hs_config.PPOAgent.num_steps

      if self.model_dir and (ppo_update_num % self.save_every == 0 or ppo_update_num == self.num_updates - 1):
        self.save_model(envs, total_num_steps)

      if ppo_update_num % hs_config.print_every == 0 and ppo_update_num > 0:
        self.print_stats(action_loss, dist_entropy, episode_rewards, ppo_update_num, start, total_num_steps, value_loss)

      if ppo_update_num % self.eval_every == 0:
        # TODO: this info should not be here!
        info = self.eval_agent(envs, info, load_env, seed)

  def eval_agent(self, envs, info, load_env, seed):
    eval_envs = make_vec_envs(load_env, seed, self.num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                              hs_config.device, allow_early_resets=True)
    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
      vec_norm.eval()
      vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    rewards = []

    def stop_eval(rews, step):
      return len(rews) >= 10

    self.gather_rollouts(None, rewards, eval_envs, exit_condition=stop_eval)

    eval_envs.close()
    print(" evaluation using {} episodes: mean reward {:.5f}\n".
          format(len(rewards), np.mean(rewards)))
    return info

  def gather_rollouts(self, rollouts, rewards, envs, exit_condition, deterministic=False):
    if rollouts is None:
      obs, _, _, infos = envs.reset()
      possible_actions = infos['possible_actions']
    else:
      obs, possible_actions = rollouts.get_observation(0)

    for step in itertools.count():
      if step == 1000:
        raise TimeoutError

      if exit_condition(rewards, step):
        break

      with torch.no_grad():
        value, action, action_log_prob = self.actor_critic(obs, possible_actions, deterministic=deterministic)

      obs, reward, done, infos = envs.step(action)
      possible_actions = infos['possible_actions']

      if rollouts is not None:
        rollouts.insert(observations=obs, actions=action, action_log_probs=action_log_prob, value_preds=value,
                        rewards=reward, not_dones=(1 - done), possible_actions=possible_actions)

      if 'end_episode_info' in infos:
        rewards.extend(i['reward'] for i in infos['end_episode_info'])

  def print_stats(self, action_loss, dist_entropy, episode_rewards, ppo_update_num, start, total_num_steps, value_loss):
    end = time.time()
    print("updates {}, num timesteps {}, fps {} \n last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
          "min/max reward {:.1f}/{:.1f}\n".
          format(ppo_update_num, total_num_steps,
                 int(total_num_steps / (end - start)),
                 len(episode_rewards),
                 np.mean(episode_rewards),
                 np.median(episode_rewards),
                 np.min(episode_rewards) if episode_rewards else float('nan'),
                 np.max(episode_rewards) if episode_rewards else float('nan'),
                 dist_entropy,
                 value_loss, action_loss))

  def save_model(self, envs, total_num_steps):
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

  def enjoy(self, make_env, checkpoint_file):
    env = make_vec_envs(make_env, hs_config.seed, num_processes=1, gamma=1, log_dir='/tmp/enjoy_log_dir',
                        device=torch.device('cpu'), allow_early_resets=False)

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

  def update(self, rollouts: RolloutStorage):
    advantages = rollouts.get_advantages()
    advantages -= advantages.mean()
    advantages /= advantages.std() + 1e-5

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
