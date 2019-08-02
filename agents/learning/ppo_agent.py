import collections
import copy
import datetime
import glob
import itertools
import os
import shutil
import tempfile
import time
import warnings
from typing import Text, Optional, Tuple

import numpy as np
import tensorboardX
import torch
import tqdm

import agents.base_agent
import game_utils
import hs_config
import shared.utils
import specs
from agents.learning.models.randomized_policy import ActorCritic
from agents.learning.shared.storage import RolloutStorage
from shared.env_utils import make_vec_envs
from shared.utils import Timer


class PPOAgent(agents.base_agent.Agent):
  def _choose(self, observation, possible_actions):
    with torch.no_grad():
      value, action, action_log_prob = self.actor_critic(observation, possible_actions, deterministic=True)
    return action

  def __init__(self, num_inputs: int, num_possible_actions: int, log_dir: str,
    experts: Tuple[agents.base_agent.Agent] = tuple()) -> None:
    assert isinstance(__class__.__name__,str)
    self.timer = Timer(__class__.__name__, verbosity=hs_config.verbosity)

    self.experiment_id = hs_config.comment
    assert specs.check_positive_type(num_possible_actions - 1, int), 'the agent can only pass'

    self.sequential_experiment_num = 0
    self._setup_logs(log_dir)
    self.log_dir = log_dir
    self.tensorboard = None
    self.envs = None
    self.eval_envs = None
    self.validation_envs = None
    self.enjoy_env = None

    self.num_inputs = num_inputs
    self.num_experts = len(experts)
    self.experts = experts
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

    self.optimizer = torch.optim.Adam(
      self.actor_critic.parameters(),
      lr=hs_config.PPOAgent.adam_lr,
    )

  def update_experiment_logging(self):
    tensorboard_dir = os.path.join(self.log_dir, r"tensorboard", "{}_{}_{}.pt".format(
      datetime.datetime.now().strftime('%b%d_%H-%M-%S'), self.experiment_id, self.sequential_experiment_num))

    if "DELETEME" in tensorboard_dir:
      tensorboard_dir = tempfile.mktemp()

    if self.tensorboard is not None:
      self.tensorboard.close()

    self.tensorboard = tensorboardX.SummaryWriter(tensorboard_dir, flush_secs=2)
    self.sequential_experiment_num += 1

  @staticmethod
  def _setup_logs(log_dir):
    try:
      os.makedirs(log_dir)
    except OSError:
      files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
      for f in files:
        os.remove(f)
    eval_log_dir = log_dir + "_eval"
    try:
      os.makedirs(eval_log_dir)
    except OSError:
      files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
      for f in files:
        os.remove(f)

  def train(self, game_manager: game_utils.GameManager, checkpoint_file: Optional[Text],
    num_updates: int = hs_config.PPOAgent.num_updates, updates_offset: int = 0) -> Tuple[Text, float, int]:
    self.update_experiment_logging()
    with self.timer("train"):
      out = self._train(game_manager, checkpoint_file, num_updates, updates_offset)
    return out

  def _train(self, game_manager: game_utils.GameManager, checkpoint_file: Optional[Text],
    num_updates: int = hs_config.PPOAgent.num_updates, updates_offset: int = 0) -> Tuple[Text, float, int]:
    assert updates_offset >= 0

    with self.timer("setup_envs"):
      envs, eval_envs, valid_envs = self.setup_envs(game_manager)

    with self.timer("load_ckpt"):
      if checkpoint_file:
        self.timer.info(f"[TRAIN] Loading ckpt {checkpoint_file}")
        self.load_checkpoint(checkpoint_file, envs)

        assert game_manager.use_heuristic_opponent is False

    assert envs.observation_space.shape == (self.num_inputs,)
    assert envs.action_space.n + self.num_experts == self.num_actions

    rollouts = RolloutStorage(self.num_inputs, self.num_actions)
    obs, _, _, info = envs.reset()
    possible_actions = self.update_possible_actions_for_expert(info)

    rollouts.store_first_transition(obs, possible_actions)
    episode_rewards = collections.deque(maxlen=10)

    start = time.time()
    total_num_steps = None
    ppo_update_num = None

    for ppo_update_num in range(updates_offset, updates_offset + num_updates):
      def stop_gathering(_, step):
        return step >= hs_config.PPOAgent.num_steps

      with self.timer("gather_rollouts"):
        self.gather_rollouts(rollouts, episode_rewards, envs, stop_gathering, False)

      with torch.no_grad():
        next_value = self.actor_critic.critic(rollouts.get_last_observation()).detach()

      with self.timer("compute_returns"):
        rollouts.compute_returns(next_value)

      with self.timer("update"):
        value_loss, action_loss, dist_entropy, policy_ratio, explained_variance = self.update(rollouts)

      rollouts.roll_over_last_transition()
      total_num_steps = ((ppo_update_num + 1) * hs_config.PPOAgent.num_processes * hs_config.PPOAgent.num_steps)
      with self.timer("print_tb"):
        self.print_stats(action_loss, dist_entropy, episode_rewards, total_num_steps, start, value_loss, policy_ratio,
                         explained_variance)

      if self.model_dir and (ppo_update_num % self.save_every == 0):
        self.save_model(envs, total_num_steps)

      good_training_performance = len(episode_rewards) == episode_rewards.maxlen and np.mean(episode_rewards) > 1 - (
        1 - hs_config.PPOAgent.winratio_cutoff) * 2
      if ppo_update_num % self.eval_every == 0 and ppo_update_num > 1 or good_training_performance:

        performance = np.mean(self.eval_agent(envs, eval_envs))
        self.tensorboard.add_scalar('dashboard/eval_performance', performance, ppo_update_num)
        if performance > hs_config.PPOAgent.winratio_cutoff:
          with self.timer("eval_agents_self_play"):
            p = self.eval_agent(envs, eval_envs)
          performance = np.mean(p)
          self.timer.info("[Train] early stopping at iteration", ppo_update_num, 'steps:', total_num_steps, performance)
          break

    checkpoint_file = self.save_model(envs, total_num_steps)

    with self.timer("eval_agents_hs"):
      outcome = self.eval_agent(envs, valid_envs)
    # import pdb; pdb.set_trace()
    test_performance = float(np.mean(outcome))

    # print("[Train] test performance", checkpoint_file, ppo_update_num, test_performance)

    # if test_performance > 0.9:
    #   self.enjoy(game_manager, checkpoint_file)

    return checkpoint_file, test_performance, ppo_update_num + 1

  def update_possible_actions_for_expert(self, info):
    possible_actions = info['possible_actions']
    if self.num_experts:
      expert_actions = torch.ones(possible_actions.shape[0], self.num_experts)
      possible_actions = torch.cat((possible_actions, expert_actions), dim=1)
    return possible_actions

  def load_checkpoint(self, checkpoint_file, envs):
    self.actor_critic, ob_rms = torch.load(checkpoint_file)
    shared.utils.get_vec_normalize(envs).ob_rms = ob_rms
    self.actor_critic.to(hs_config.device)
    self.optimizer = torch.optim.Adam(
      self.actor_critic.parameters(),
      lr=hs_config.PPOAgent.adam_lr,
    )

  def setup_envs(self, game_manager: game_utils.GameManager):

    if self.envs is None:
      print("[Train] Loading training environments")
      game_manager.use_heuristic_opponent = False
      self.envs = make_vec_envs(game_manager, self.num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                                hs_config.device, allow_early_resets=False)
    else:
      game_manager.use_heuristic_opponent = False
      self.get_last_env(self.envs).set_opponents(opponents=game_manager.opponents,
                                                 opponent_obs_rmss=game_manager.opponent_normalization_factors)

    if self.eval_envs is None:
      print("[Train] Loading eval environments")
      game_manager.use_heuristic_opponent = False
      self.eval_envs = make_vec_envs(game_manager, self.num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                                     hs_config.device, allow_early_resets=True)
    else:
      game_manager.use_heuristic_opponent = False
      self.get_last_env(self.eval_envs).set_opponents(opponents=game_manager.opponents,
                                                      opponent_obs_rmss=game_manager.opponent_normalization_factors)

    if self.validation_envs is None:
      print("[Train] Loading validation environments")
      game_manager.use_heuristic_opponent = True
      self.validation_envs = make_vec_envs(game_manager, self.num_processes, hs_config.PPOAgent.gamma, self.log_dir,
                                           hs_config.device, allow_early_resets=True)
    # else:
    #   self.get_last_env(self.validation_envs).set_opponents(opponents=game_manager.opponents,
    #                                                         opponent_obs_rmss=game_manager.opponent_normalization_factors)

    shared.utils.get_vec_normalize(self.envs).ob_rms = None
    shared.utils.get_vec_normalize(self.eval_envs).ob_rms = None
    shared.utils.get_vec_normalize(self.validation_envs).ob_rms = None
    # assert shared.utils.get_vec_normalize(self.envs).ob_rms is not None
    # assert shared.utils.get_vec_normalize(self.eval_envs).ob_rms is not None
    # assert shared.utils.get_vec_normalize(self.validation_envs).ob_rms is not None
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

  def eval_agent(self, train_envs, eval_envs):
    vec_norm = shared.utils.get_vec_normalize(eval_envs)
    assert vec_norm is not None
    vec_norm.eval()
    vec_norm.ob_rms = shared.utils.get_vec_normalize(train_envs).ob_rms

    rewards = []

    def stop_eval(rews, step):
      return len(rews) >= hs_config.PPOAgent.num_eval_games

    self.gather_rollouts(None, rewards, eval_envs, exit_condition=stop_eval)
    return rewards

  def gather_rollouts(self, rollouts, rewards, envs, exit_condition, deterministic=False, game_stats=None):

    if rollouts is None:
      obs, _, _, infos = envs.reset()
      possible_actions = self.update_possible_actions_for_expert(infos)
    else:
      obs, possible_actions = rollouts.get_observation(0)

    for step in itertools.count():
      if step == 10000:
        raise TimeoutError

      if exit_condition(rewards, step):
        break

      with torch.no_grad():
        value, action, action_log_prob = self.actor_critic(obs, possible_actions, deterministic=deterministic)

      self.maybe_render(action, envs)
      # could be parallelized
      # for idx, a in enumerate(action):
      #   if a >= envs.action_space.n:
      #     expert_number = action[idx] - envs.action_space.n
      #     with torch.no_grad():
      #       _, expert_action, _ = self.experts[expert_number].act(
      #         obs[idx:idx + 1, :],
      #         possible_actions[idx: idx + 1, :envs.action_space.n], deterministic=deterministic)
      #       action[idx, 0] = expert_action

      with self.timer("agent_step"):
        obs, reward, done, infos = envs.step(action)
      assert not all(done) or all(r in (-1., -.1) for r in infos['reward'])
      #assert not done and infos['reward'][0] == 0

      possible_actions = self.update_possible_actions_for_expert(infos)

      if rollouts is not None:
        rollouts.insert(observations=obs, actions=action, action_log_probs=action_log_prob, value_preds=value,
                        rewards=reward, not_dones=(1 - done), possible_actions=possible_actions)
      assert 'game_statistics' in specs.OPTIONAL_INFO_KEYS
      if 'game_statistics' in infos:
        warnings.warn('breaks if reward shaping')
        rewards.extend(i[1] for i in infos['game_statistics'] if i[1] != 0)

  def maybe_render(self, action, envs):
    if hs_config.Environment.render_after_step:
      act = envs.vectorized_env.vectorized_env.remotes[0].render()
      try:
        act = int(chr(act))
      except ValueError:
        pass
      else:
        action[0] = act

  def print_stats(self, action_loss, dist_entropy, episode_rewards, time_step, start, value_loss, policy_ratio,
    explained_variance):
    end = time.time()
    if episode_rewards:
      fps = int(time_step / (end - start))
      if fps > 1000:
        fps = float('nan')

      self.tensorboard.add_scalar('zdebug/steps_per_second', fps, time_step)
      self.tensorboard.add_scalar('dashboard/mean_reward', np.mean(episode_rewards), time_step)

    self.tensorboard.add_scalar('train/entropy', dist_entropy, time_step)
    self.tensorboard.add_scalar('train/value_loss', value_loss, time_step)
    self.tensorboard.add_scalar('train/action_loss', action_loss, time_step)
    # too low, no learning, fix batch size
    self.tensorboard.add_scalar('train/policy_ratio', policy_ratio, time_step)

    #  >= 1 good ev; =< 0 null predictor
    self.tensorboard.add_scalar('train/explained_variance', explained_variance, time_step)

    self.tensorboard.add_scalar('zlosses/entropy', dist_entropy * self.entropy_coeff, time_step)
    self.tensorboard.add_scalar('zlosses/value_loss', value_loss * self.value_loss_coeff, time_step)
    self.tensorboard.add_scalar('zlosses/action_loss', action_loss, time_step)

  def save_model(self, envs, total_num_steps):
    try:
      os.makedirs(hs_config.PPOAgent.save_dir)
    except OSError:
      pass
    # a really ugly way to save a model to cpu
    if hs_config.use_gpu:
      model = copy.deepcopy(self.actor_critic).cpu()
    else:
      model = self.actor_critic

    # save_model = (self.actor_critic.state_dict(), self.optimizer, get_vec_normalize(envs).ob_rms)
    save_model = (model, shared.utils.get_vec_normalize(envs).ob_rms)
    assert shared.utils.get_vec_normalize(self.envs).ob_rms is None
    assert shared.utils.get_vec_normalize(self.eval_envs).ob_rms is None
    assert shared.utils.get_vec_normalize(self.validation_envs).ob_rms is None

    checkpoint_name = f"id={self.experiment_id}:steps={total_num_steps}:inputs={self.num_inputs}.pt"
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
      self.enjoy_env = make_vec_envs(game_manager, num_processes=1, gamma=0, log_dir=None, device=hs_config.device,
                                     allow_early_resets=True)
    else:
      self.get_last_env(self.enjoy_env).set_opponent(opponents=game_manager.opponents,
                                                     opponent_obs_rmss=game_manager.opponent_normalization_factors)

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
        self.enjoy_env.reset()
        # print('DONE')
        # self.enjoy_env.close()
        # break

  def update(self, rollouts: RolloutStorage):
    advantages = rollouts.get_advantages()
    advantages -= advantages.mean()
    advantages /= advantages.std() + 1e-5

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0
    values_pred = []
    values_gt = []
    ratio_epoch = 0

    for e in range(self.num_ppo_epochs):
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
        (value_loss * self.value_loss_coeff + action_loss - dist_entropy * self.entropy_coeff).backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        value_loss_epoch += value_loss.item()
        action_loss_epoch += action_loss.item()
        dist_entropy_epoch += dist_entropy.item()

        values_pred.extend(value_preds_batch)
        values_gt.extend(values)

        ratio_epoch += ratio.mean().item()

    num_updates = self.num_ppo_epochs * self.num_mini_batch

    values_pred = torch.tensor(values_pred)
    values_gt = torch.tensor(values_gt)

    explained_variance_epoch = ((1 - (values_pred - values_gt).var()) / values_pred.var()).item()

    value_loss_epoch /= num_updates
    action_loss_epoch /= num_updates
    dist_entropy_epoch /= num_updates
    ratio_epoch /= num_updates
    explained_variance_epoch /= num_updates

    return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, ratio_epoch, explained_variance_epoch

  def self_play(self, game_manager: game_utils.GameManager, checkpoint_file):
    self.update_experiment_logging()

    # if checkpoint_file is None:
    #   checkpoint_file = self.get_latest_checkpoint_file()

    # https://openai.com/blog/openai-five/
    updates_so_far = 0
    updates_schedule = [1, ]
    updates_schedule.extend([hs_config.PPOAgent.num_updates, ] * hs_config.SelfPlay.num_opponent_updates)
    old_win_ratio = -1
    pbar = tqdm.tqdm(total=sum(updates_schedule))
    try:
      for self_play_iter, num_updates in enumerate(updates_schedule):
        print("Iter", self_play_iter, 'old_win_ratio', old_win_ratio, 'updates_so_far', updates_so_far)
        new_checkpoint_file, win_ratio, updates_so_far = self._train(
          game_manager, checkpoint_file=checkpoint_file, num_updates=num_updates, updates_offset=updates_so_far)
        assert game_manager.use_heuristic_opponent is False or self_play_iter == 0

        self.tensorboard.add_scalar('dashboard/heuristic_latest', win_ratio, self_play_iter)
        if win_ratio >= old_win_ratio:
          print('updating checkpoint')
          checkpoint_file = new_checkpoint_file
          shutil.copyfile(checkpoint_file, checkpoint_file + "_iter_" + str(self_play_iter))
          self.tensorboard.add_scalar('winning_ratios/heuristic_best', win_ratio, self_play_iter)
          old_win_ratio = win_ratio
          assert not game_manager.use_heuristic_opponent or self_play_iter == 0

        assert game_manager.use_heuristic_opponent is False or self_play_iter == 0
        game_manager.add_learning_opponent(checkpoint_file)

        #self.envs.vectorized_env.vectorized_env.envs[0].print_nash()
        # self.actor_critic.reset_actor()
        self.optimizer.state = collections.defaultdict(dict)  # Reset state
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


class Expert(agents.base_agent.Agent):
  def _choose(self, *args, **kwargs):
    raise NotImplementedError

  def __init__(self, checkpoint_file: str) -> None:
    self.actor_critic, ob_rms = torch.load(checkpoint_file)
    self.actor_critic.to(hs_config.device)
    # shared.utils.get_vec_normalize(envs).ob_rms = ob_rms

  def act(self, obs, possible_actions, deterministic):
    # p1, p2 = obs[:, :55], obs[:, 55:]
    # p1 = p1[:, 30:]  # hide player's hand
    # p2 = p2[:, 30:]  # hide opponent's hand
    # obs = torch.cat((p1, p2), dim=1)
    return self.actor_critic(obs, possible_actions, deterministic=deterministic)
