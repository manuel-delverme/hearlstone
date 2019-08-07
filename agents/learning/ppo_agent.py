import collections
import copy
import datetime
import glob
import itertools
import os
import tempfile
import time
from typing import Text, Optional, Tuple, List, Callable

import numpy as np
import tensorboardX
import torch
import tqdm

import agents.base_agent
import game_utils
import hs_config
import specs
from agents.learning.models.randomized_policy import ActorCritic
from agents.learning.shared.storage import RolloutStorage
from shared.env_utils import make_vec_envs
from shared.utils import HSLogger


def get_grad_norm(model):
  total_norm = 0
  for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
  total_norm = total_norm ** (1. / 2)  # SUPER SLOW
  return total_norm


class PPOAgent(agents.base_agent.Agent):
  def _choose(self, observation, possible_actions):
    with torch.no_grad():
      value, action, action_log_prob = self.actor_critic(observation, possible_actions, deterministic=True)
    return action

  def __init__(self, num_inputs: int, num_possible_actions: int) -> None:
    assert isinstance(__class__.__name__, str)
    self.timer = HSLogger(__class__.__name__, log_to_stdout=hs_config.log_to_stdout)

    self.experiment_id = hs_config.comment
    assert specs.check_positive_type(num_possible_actions - 1, int), 'the agent can only pass'

    self.tensorboard = None
    self.envs = None
    self.eval_envs = None
    self.validation_envs = None
    self.enjoy_env = None

    self.num_inputs = num_inputs
    self.num_actions = num_possible_actions

    actor_critic_network = ActorCritic(self.num_inputs, self.num_actions)
    actor_critic_network.to(hs_config.device)
    self.actor_critic = actor_critic_network

    self.clip_param = hs_config.PPOAgent.clip_epsilon
    assert specs.check_positive_type(self.clip_param, float)

    self.num_ppo_epochs = hs_config.PPOAgent.ppo_epoch
    assert specs.check_positive_type(self.num_ppo_epochs, int)

    self.num_mini_batch = hs_config.PPOAgent.num_mini_batches
    assert specs.check_positive_type(self.num_mini_batch, int)

    self.value_loss_coeff = hs_config.PPOAgent.value_loss_coeff
    assert specs.check_positive_type(self.value_loss_coeff, float)

    self.entropy_coeff = hs_config.PPOAgent.entropy_coeff
    assert specs.check_positive_type(self.entropy_coeff, float)

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

    self.pi_optimizer = torch.optim.Adam(
        self.actor_critic.actor.parameters(),
        lr=hs_config.PPOAgent.adam_lr,
    )
    self.value_optimizer = torch.optim.Adam(
        self.actor_critic.critic.parameters(),
        lr=hs_config.PPOAgent.adam_lr,
    )

  def update_experiment_logging(self):
    tensorboard_dir = os.path.join(f"logs/tensorboard/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{self.experiment_id}.pt")

    if "DELETEME" in tensorboard_dir:
      tensorboard_dir = tempfile.mktemp()

    if self.tensorboard is not None:
      self.tensorboard.close()

    self.tensorboard = tensorboardX.SummaryWriter(tensorboard_dir, flush_secs=2)

  def train(self, game_manager: game_utils.GameManager, checkpoint_file: Optional[Text],
      num_updates: int = hs_config.PPOAgent.num_updates, updates_offset: int = 0) -> Tuple[Text, float, int]:
    assert updates_offset >= 0

    with self.timer("setup_envs"):
      envs, eval_envs, valid_envs = self.setup_envs(game_manager)

    with self.timer("load_ckpt"):
      if checkpoint_file:
        print(f"[Train] Loading ckpt {checkpoint_file}")
        self.load_checkpoint(checkpoint_file, envs)

        assert game_manager.use_heuristic_opponent is False or num_updates == 1

    assert envs.observation_space.shape == (self.num_inputs,)
    assert envs.action_space.n == self.num_actions

    rollouts = RolloutStorage(self.num_inputs, self.num_actions)
    obs, _, _, info = envs.reset()
    possible_actions = info['possible_actions']

    rollouts.store_first_transition(obs, possible_actions)
    episode_rewards = collections.deque(maxlen=hs_config.PPOAgent.num_episodes_for_early_exit)

    start = time.time()  # TODO: bugged
    total_num_steps = None
    ppo_update_num = None

    for ppo_update_num in tqdm.tqdm(range(updates_offset, updates_offset + num_updates), position=1, desc='train'):
      def stop_gathering(_, step):
        return step >= hs_config.PPOAgent.num_steps

      with self.timer("gather_rollouts"):
        self.gather_rollouts(rollouts, episode_rewards, envs, stop_gathering, False)

      with torch.no_grad():
        next_value = self.actor_critic.critic(rollouts.get_last_observation()).detach()

      with self.timer("compute_returns"):
        rollouts.compute_returns(next_value)

      with self.timer("update"):
        value_loss, action_loss, dist_entropy, policy_ratio, mean_value, grad_pi, grad_value = self.update(rollouts)

      rollouts.roll_over_last_transition()
      total_num_steps = ((ppo_update_num + 1) * hs_config.PPOAgent.num_processes * hs_config.PPOAgent.num_steps)
      with self.timer("print_tb"):
        self.print_stats(action_loss, dist_entropy, episode_rewards, total_num_steps, start, value_loss, policy_ratio,
                         mean_value, grad_pi=grad_pi, grad_value=grad_value)

      if ppo_update_num > 1:
        if self.model_dir and (ppo_update_num % self.save_every == 0):
          self.save_model(total_num_steps)

        if ppo_update_num % self.eval_every == 0:
          with self.timer("eval_agents_self_play"):
            _rewards, _scores = self.eval_agent(eval_envs)
          performance = np.mean(_rewards)
          self.tensorboard.add_scalar('dashboard/eval_performance', performance, ppo_update_num)

          if performance > hs_config.PPOAgent.performance_to_early_exit:
            self.timer.info("[Train] early stopping at iteration", ppo_update_num, 'steps:', total_num_steps, performance)
            print("[Train] early stopping at iteration", ppo_update_num, 'steps:', total_num_steps, performance)
            break

    checkpoint_file = self.save_model(total_num_steps)

    with self.timer("eval_agents_hs"):
      rewards, outcomes = self.eval_agent(valid_envs)

    test_performance = float(np.mean(rewards))
    return checkpoint_file, test_performance, ppo_update_num + 1

  def load_checkpoint(self, checkpoint_file, envs):
    self.actor_critic, = torch.load(checkpoint_file)
    self.actor_critic.to(hs_config.device)
    self.pi_optimizer = torch.optim.Adam(
        self.actor_critic.parameters(),
        lr=hs_config.PPOAgent.adam_lr,
    )
    self.value_optimizer = torch.optim.Adam(
        self.actor_critic.parameters(),
        lr=hs_config.PPOAgent.adam_lr, )

  def setup_envs(self, game_manager: game_utils.GameManager):
    if self.envs is None:
      print("[Train] Loading training environments")
      # TODO @d3sm0 clean this up
      game_manager.use_heuristic_opponent = False
      self.envs = make_vec_envs(game_manager, self.num_processes)
    else:
      game_manager.use_heuristic_opponent = False
      self.get_last_env(self.envs).set_opponents(opponents=game_manager.opponents)

    if self.eval_envs is None:
      print("[Train] Loading eval environments")
      game_manager.use_heuristic_opponent = False
      self.eval_envs = make_vec_envs(game_manager, self.num_processes)
    else:
      game_manager.use_heuristic_opponent = False
      self.get_last_env(self.eval_envs).set_opponents(opponents=game_manager.opponents)

    if self.validation_envs is None:
      print("[Train] Loading validation environments")
      game_manager.use_heuristic_opponent = True
      self.validation_envs = make_vec_envs(game_manager, self.num_processes)

    return self.envs, self.eval_envs, self.validation_envs

  def get_last_env(self, env):
    for _ in range(256):
      if hasattr(env, 'vectorized_env'):
        env = env.vectorized_env
      else:
        break
    else:
      raise AttributeError('Infinite loop')

    return env

  def eval_agent(self, eval_envs):
    rewards = []

    def stop_eval(rews, step):
      return len(rews) >= hs_config.PPOAgent.num_eval_games

    opponents = []
    self.gather_rollouts(None, rewards, eval_envs, exit_condition=stop_eval, opponents=opponents)

    scores = collections.defaultdict(list)
    for k, r in zip(opponents, rewards):
      scores[k].append(r)
    return rewards, dict(scores)

  def gather_rollouts(self, rollouts, rewards: List, envs, exit_condition: Callable[[List, int], bool], deterministic: bool = False,
      opponents: list = None):
    if rollouts is None:
      obs, _, _, infos = envs.reset()
      possible_actions = infos['possible_actions']
    else:
      obs, possible_actions = rollouts.get_observation(0)

    for step in itertools.count():
      if step == 10000:
        raise TimeoutError

      if exit_condition(rewards, step):
        break

      with torch.no_grad():
        value, action, action_log_prob = self.actor_critic(obs, possible_actions, deterministic=deterministic)

      with self.timer("agent_step"):
        obs, reward, done, infos = envs.step(action)
      assert all((not _done) or (_reward in (-1., 1)) for _done, _reward in zip(done, rewards))

      possible_actions = infos['possible_actions']

      if rollouts is not None:
        rollouts.insert(observations=obs, actions=action, action_log_probs=action_log_prob, value_preds=value,
                        rewards=reward, not_dones=(1 - done), possible_actions=possible_actions)
      assert 'game_statistics' in specs.TERMINAL_GAME_INFO_KEYS
      if 'game_statistics' in infos:

        for info in infos['game_statistics']:
          outcome = info['outcome']
          if outcome != 0:
            assert isinstance(outcome, np.float32)
            rewards.append(outcome)
            if opponents is not None:
              opponents.append(info['opponent_nr'])

  def print_stats(self, action_loss, dist_entropy, episode_rewards, time_step, start, value_loss, policy_ratio, mean_value,
      grad_value, grad_pi):
    end = time.time()
    if episode_rewards:
      fps = int(time_step / (end - start))
      if fps > 1000:
        fps = float('nan')

      self.tensorboard.add_scalar('zdebug/steps_per_second', fps, time_step)
      self.tensorboard.add_scalar('dashboard/mean_reward', np.mean(episode_rewards) / 2 + 0.5, time_step)

    self.tensorboard.add_scalar('train/grad_value', grad_value, time_step)
    self.tensorboard.add_scalar('train/grad_pi', grad_pi, time_step)
    self.tensorboard.add_scalar('train/entropy', dist_entropy, time_step)
    self.tensorboard.add_scalar('train/value_loss', value_loss, time_step)
    self.tensorboard.add_scalar('train/action_loss', action_loss, time_step)
    # too low, no learning, fix batch size
    self.tensorboard.add_scalar('train/policy_ratio', policy_ratio, time_step)
    self.tensorboard.add_scalar('train/mean_value', mean_value, time_step)

    self.tensorboard.add_scalar('zlosses/entropy', dist_entropy * self.entropy_coeff, time_step)
    self.tensorboard.add_scalar('zlosses/value_loss', value_loss * self.value_loss_coeff, time_step)
    self.tensorboard.add_scalar('zlosses/action_loss', action_loss, time_step)

  def save_model(self, total_num_steps):
    try:
      os.makedirs(hs_config.PPOAgent.save_dir)
    except OSError:
      pass
    # a really ugly way to save a model to cpu
    if hs_config.use_gpu:
      model = copy.deepcopy(self.actor_critic).cpu()
    else:
      model = self.actor_critic

    save_model = (model,)
    checkpoint_name = f"id={self.experiment_id}:steps={total_num_steps}.pt"
    checkpoint_file = os.path.join(hs_config.PPOAgent.save_dir, checkpoint_name)
    torch.save(save_model, checkpoint_file)
    return checkpoint_file

  def get_latest_checkpoint_file(self):
    checkpoint_name = "{}:{}.pt".format(self.experiment_id, "*")
    checkpoint_files = os.path.join(hs_config.PPOAgent.save_dir, checkpoint_name)
    checkpoints = glob.glob(checkpoint_files)

    if not checkpoints:
      return None

    ids = {}
    for file_name in checkpoints:
      x = file_name.replace(":", "-")
      last_part = x.split("-")[-1]
      num_iters = last_part[:-3]
      ids[file_name] = int(num_iters)

    checkpoints = sorted(checkpoints, key=lambda xi: ids[xi])
    latest_checkpoint = checkpoints[-1]
    return latest_checkpoint

  def enjoy(self, game_manager: game_utils.GameManager, checkpoint_file):
    if self.enjoy_env is None:
      print("[Train] Loading training environments")
      self.enjoy_env = make_vec_envs(game_manager, num_processes=1, device=hs_config.device)
    else:
      self.get_last_env(self.enjoy_env).set_opponent(opponents=game_manager.opponents)

    # We need to use the same statistics for normalization as used in training
    if checkpoint_file:
      self.load_checkpoint(checkpoint_file, self.enjoy_env)

    obs, _, _, infos = self.enjoy_env.reset()

    while True:
      with torch.no_grad():
        value, action, _ = self.actor_critic(obs, infos['possible_actions'], deterministic=True)
        action_distribution, value = self.actor_critic.actor_critic(obs, infos['possible_actions'])

      self.enjoy_env.render(choice=action, action_distribution=action_distribution, value=value)
      obs, reward, done, infos = self.enjoy_env.step(action)

      if done:
        self.enjoy_env.render(choice=action, action_distribution=action_distribution, value=value, reward=reward)
        self.enjoy_env.reset()

  def update(self, rollouts: RolloutStorage):
    advantages = rollouts.get_advantages()
    advantages -= advantages.mean()
    advantages /= advantages.std() + 1e-5

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0
    grad_norm_pi_epoch = 0
    grad_norm_value_epoch = 0
    values_pred = []
    values_gt = []
    ratio_epoch = 0

    for e in range(self.num_ppo_epochs):
      data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

      for sample in data_generator:
        (obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ,
         possible_actions) = sample

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch, possible_actions)

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

        self.pi_optimizer.zero_grad()
        (action_loss - dist_entropy * self.entropy_coeff).backward()
        grad_norm_pi = get_grad_norm(self.actor_critic.actor)  # SUPER SLOW
        self.pi_optimizer.step()

        #torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)

        self.value_optimizer.zero_grad()
        (value_loss * self.value_loss_coeff).backward()

        grad_norm_critic = get_grad_norm(self.actor_critic.critic)  # SUPER SLOW
        torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)

        self.value_optimizer.step()
        grad_norm_pi_epoch += grad_norm_pi
        grad_norm_value_epoch += grad_norm_critic
        value_loss_epoch += value_loss.item()
        action_loss_epoch += action_loss.item()
        dist_entropy_epoch += dist_entropy.item()

        values_pred.extend(value_preds_batch)
        values_gt.extend(values)

        ratio_epoch += ratio.mean().item()

    num_updates = self.num_ppo_epochs * self.num_mini_batch

    values_pred = torch.tensor(values_pred)

    value_loss_epoch /= num_updates
    action_loss_epoch /= num_updates
    dist_entropy_epoch /= num_updates
    ratio_epoch /= num_updates
    value_mean = torch.mean(values_pred)

    return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, ratio_epoch, value_mean, grad_norm_pi_epoch, grad_norm_value_epoch

  def self_play(self, game_manager: game_utils.GameManager, checkpoint_file):
    self.update_experiment_logging()
    updates_so_far = 0
    updates_schedule = [1, ]
    updates_schedule.extend([hs_config.PPOAgent.num_updates, ] * hs_config.SelfPlay.num_opponent_updates)
    old_win_ratio = -1  # TODO: this assumes the checkpoint has worse performance than the 1iteration ppo update
    pbar = tqdm.tqdm(total=sum(updates_schedule), desc='self-play')
    try:
      for self_play_iter, num_updates in enumerate(updates_schedule):
        print("Iter", self_play_iter, 'old_win_ratio', old_win_ratio, 'updates_so_far', updates_so_far)

        new_checkpoint_file, win_ratio, updates_so_far = self.train(game_manager, checkpoint_file, num_updates, updates_so_far)
        assert game_manager.use_heuristic_opponent is False or self_play_iter == 0

        self.tensorboard.add_scalar('dashboard/heuristic_latest', win_ratio / 2 + 0.5, self_play_iter)
        self.tensorboard.add_scalar('dashboard/self_play_iter', self_play_iter, updates_so_far)

        checkpoint_file = new_checkpoint_file
        # if win_ratio >= old_win_ratio:
        #   print('updating checkpoint')
        #   checkpoint_file = new_checkpoint_file
        #   shutil.copyfile(checkpoint_file, checkpoint_file + "_iter_" + str(self_play_iter))
        #   self.tensorboard.add_scalar('winning_ratios/heuristic_best', win_ratio / 2 + 0.5, self_play_iter)
        #   old_win_ratio = win_ratio
        #   assert not game_manager._use_heuristic_opponent

        game_manager.add_learned_opponent(checkpoint_file)  # TODO: avoid adding the same player
        # self.pi_optimizer.state = collections.defaultdict(dict)  # Reset state
        # self.value_optimizer.state = collections.defaultdict(dict)  # Reset state

        pbar.update(num_updates)

    except KeyboardInterrupt:
      print("Captured KeyboardInterrupt from user, quitting")
      if self.eval_envs:
        self.eval_envs.close()
      if self.envs:
        self.envs.close()
      if self.validation_envs:
        self.validation_envs.close()
      self.tensorboard.close()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.tensorboard.close()
