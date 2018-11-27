import tensorboardX
import config
import copy
import numpy as np

import torch
import agents.base_agent
import random
from typing import Tuple, List

import agents.learning.replay_buffers
from agents.learning import shared
from agents.learning.models import dqn
# from agents.learning.models import double_q_learning
import tqdm
import config
from torch.autograd import Variable
import os
import time
import torch.nn
import subprocess

USE_CUDA = False


class DQNAgent(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions, should_flip_board=False,
               model_path="checkpoints/checkpoint.pth.tar", record=True) -> None:
    self.use_double_q = config.DQNAgent.use_double_q
    self.use_target = config.DQNAgent.use_target
    assert not (self.use_double_q and self.use_target)

    self.num_actions = num_actions
    self.num_inputs = num_inputs
    self.gamma = config.DQNAgent.gamma
    self.batch_size = config.DQNAgent.batch_size
    self.warmup_steps = config.DQNAgent.warmup_steps
    self.model_path = model_path

    self.q_network = dqn.DQN(num_inputs, num_actions, USE_CUDA)
    self.q_network.build_network()

    if self.use_target or self.use_double_q:
      self.q_network_target = copy.deepcopy(self.q_network)
      self.q_network_target.build_network()

    optimizer = config.DQNAgent.optimizer

    self.optimizer = optimizer(
      self.q_network.parameters(),
      lr=config.DQNAgent.lr,
      weight_decay=config.DQNAgent.l2_decay,
    )
    self.replay_buffer = agents.learning.replay_buffers.PrioritizedBufferOpenAI(
      config.DQNAgent.buffer_size, num_inputs, num_actions
    )
    self.loss = torch.nn.SmoothL1Loss(reduction='none')
    experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = 'runs/{}'.format(experiment_name)

    if record:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir)
      print("Experiment name:", experiment_name)
      cmd = "find {} -name '*.py' | grep -v venv | tar -cvf {}/code.tar --files-from -".format(os.getcwd(), log_dir)
      subprocess.check_output(cmd, shell=True)
    else:
      self.summary_writer = tensorboardX.SummaryWriter(log_dir='/tmp/trash/')


  def load_model(self, model_path=None):
    if model_path is None:
      model_path = self.model_path
    self.q_network.load_state_dict(torch.load(model_path))
    print('loaded', model_path)

  def train_step(self, states, actions, rewards, next_states, dones, next_possible_actionss, indices, weights):
    state_action_pairs = np.concatenate((states, actions), axis=1)
    del states, actions

    not_done_mask = dones == 0

    if self.use_target:
      action_selection_network = self.q_network_target
      q_value_network = self.q_network_target
    elif self.use_double_q:
      action_selection_network = self.q_network
      q_value_network = self.q_network_target
    else:
      action_selection_network = self.q_network
      q_value_network = self.q_network

    # TODO: remove loops
    best_future_actions = np.empty(shape=(self.batch_size, self.num_actions))
    for idx, (state, possible_actions) in enumerate(zip(next_states, next_possible_actionss)):
      q_values = self.get_q_values(action_selection_network, state, possible_actions)
      best_future_action_idx = torch.argmax(q_values.detach())
      best_future_actions[idx] = possible_actions[best_future_action_idx]

    next_best_state_action_pairs = np.concatenate((next_states, best_future_actions), axis=1)
    target_future_q_values = q_value_network(next_best_state_action_pairs)
    target_future_q_values = target_future_q_values.detach().squeeze()

    # what the q_network should have estimated
    rewards = torch.FloatTensor(rewards)

    expected_q_values = rewards + not_done_mask * (self.gamma * target_future_q_values)

    # what the q_network estimates
    q_values = self.q_network(state_action_pairs).squeeze()

    # expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
    weights = torch.FloatTensor(weights)
    # loss = (q_values - expected_q_values).pow(2) * weights
    loss = self.loss(q_values, expected_q_values) #  * weights
    assert loss.shape == (self.batch_size, )

    priorities = loss * weights + 1e-5
    loss = loss.mean()

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.1)
    # for n, p in filter(lambda np: np[1].grad is not None, self.model.named_parameters()):
    # self.summary_writer.add_histogram('grad.' + n, p.grad.data.cpu().numpy(), global_step=step_nr)
    # self.summary_writer.add_histogram(n, p.data.cpu().numpy(), global_step=step_nr)
    self.replay_buffer.update_priorities(indices, priorities.data.cpu().numpy())
    self.optimizer.step()
    return loss

  def render(self, env):
    observation, reward, terminal, info = env.reset()
    while True:

      possible_actions = info['possible_actions']
      q_values = self.get_q_values(self.q_network, observation, possible_actions).detach().numpy()
      best_action = np.argmax(q_values)
      action = possible_actions[best_action]

      print('='*100)
      for a, q in zip(info['original_info']['possible_actions'], q_values):
        print(a, q)
      print(env.render(info={}))

      observation, reward, done, info = env.step(action)

      if done:
        print(done, reward)
        observation, reward, terminal, info = env.reset()

  def train(self, env, game_steps=None, checkpoint_every=10000, target_update_every=config.DQNAgent.target_update, ):
    if game_steps is None:
      game_steps = config.DQNAgent.training_steps
    observation, reward, terminal, info = env.reset()
    epsilon_schedule = shared.epsilon_schedule(
      offset=config.DQNAgent.warmup_steps,
      epsilon_decay=config.DQNAgent.epsilon_decay
    )
    beta_schedule = shared.epsilon_schedule(
      offset=config.DQNAgent.warmup_steps,
      epsilon_decay=config.DQNAgent.beta_decay,
    )

    iteration_params = zip(range(game_steps), epsilon_schedule, beta_schedule)

    for step_nr, epsilon, beta in tqdm.tqdm(iteration_params, total=game_steps):
      # self.summary_writer.add_histogram('observation', observation, global_step=step_nr)

      action = self.act(observation, info['possible_actions'], epsilon, step_nr=step_nr)

      next_observation, reward, done, info = env.step(action)
      self.learn_from_experience(observation, action, reward, next_observation, done, info['possible_actions'], step_nr,
                                 beta)

      observation = next_observation

      self.summary_writer.add_scalar('dqn/epsilon', epsilon, step_nr)
      self.summary_writer.add_scalar('dqn/beta', beta, step_nr)

      if done:
        game_value = env.game_value()
        self.summary_writer.add_scalar('game_stats/end_turn', env.simulation.game.turn, step_nr)
        self.summary_writer.add_scalar('game_stats/game_value', (game_value + 1) / 2, step_nr)

        assert reward in (-1.0, 0.0, 1.0)
        observation, reward, terminal, info = env.reset()
      else:
        assert abs(reward) < 1

      if (self.use_double_q or self.use_target) and step_nr % target_update_every == 0:
        shared.sync_target(self.q_network, self.q_network_target)
      if step_nr % checkpoint_every == 0 and step_nr > 0:
        torch.save(self.q_network.state_dict(), self.model_path)

  def choose(self, observation, info):
    board_center = observation.shape[1] // 2
    if self.should_flip_board:
      observation = np.concatenate(observation[board_center:], observation[:board_center], axis=1)

    action = self.q_network.act(observation, info['possible_actions'], 0.0)
    return action

  def learn_from_experience(self, observation, action, reward, next_state, done, next_actions, step_nr, beta):
    action = np.array(action)
    reward = np.array(reward)
    done = np.array(done)
    self.replay_buffer.push(observation, action, reward, next_state, done, next_actions)
    if len(self.replay_buffer) > max(self.batch_size, self.warmup_steps):
      state, action, reward, next_state, done, next_actions, indices, weights = self.replay_buffer.sample(
        self.batch_size, beta)

      for _ in range(config.DQNAgent.nr_epochs):
        loss = self.train_step(state, action, reward, next_state, done, next_actions, indices, weights)
      self.summary_writer.add_scalar('dqn/loss', loss, step_nr)

  def act(self, state: np.array, possible_actions: List[Tuple[int, int]], epsilon: float, step_nr: int = None):
    assert isinstance(state, np.ndarray)
    assert isinstance(possible_actions, tuple)
    assert isinstance(possible_actions[0], tuple)
    assert isinstance(possible_actions[0][0], int)
    assert isinstance(epsilon, float)

    if random.random() > epsilon:
      q_values = self.get_q_values(self.q_network, state, possible_actions)
      if step_nr is not None:
        self.summary_writer.add_scalar('dqn/minq', min(q_values), step_nr)
        self.summary_writer.add_scalar('dqn/maxq', max(q_values), step_nr)

      best_action = torch.argmax(q_values)
      action = possible_actions[best_action.detach()]
    else:
      action, = random.sample(possible_actions, 1)
    return action

  @staticmethod
  def get_q_values(q_network, state, possible_actions):
    state_action_pairs = []
    for possible_action in possible_actions:
      state_action_pair = np.append(state, possible_action)
      state_action_pairs.append(state_action_pair)
    q_values = q_network(state_action_pairs)
    return q_values

  def __del__(self):
    self.summary_writer.close()


class DQNAgentBaselines(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions) -> None:
    import baselines.deepq
    self.learn = baselines.deepq.learn
    self.num_actions = num_actions
    self.num_inputs = num_inputs

    experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
    log_dir = 'runs/baselines_{}'.format(experiment_name)
    self.summary_writer = tensorboardX.SummaryWriter(log_dir=log_dir)
    print("Experiment name:", experiment_name)
    cmd = "find {} -name '*.py' | grep -v venv | tar -cvf {}/code.tar --files-from -".format(os.getcwd(), log_dir)
    subprocess.check_output(cmd, shell=True)

  def train(self, env, game_steps=None, checkpoint_every=10000, target_update_every=100, ):
    if game_steps is None:
      game_steps = config.DQNAgent.training_steps

    model = self.learn(
      env=env,
      total_timesteps=game_steps,
      network='mlp',
    )

  def _learn(self):
    long_term = min(game_steps / 6, int(2e4))
    epsilon_schedule = shared.epsilon_schedule(epsilon_decay=long_term)
    beta_schedule = shared.epsilon_schedule(epsilon_decay=long_term)

    iteration_params = zip(range(game_steps), epsilon_schedule, beta_schedule)

    for step_nr, epsilon, beta in tqdm.tqdm(iteration_params, total=game_steps):

      action = self.act(observation, info['possible_actions'], epsilon, step_nr=step_nr)

      next_observation, reward, done, info = env.step(action)
      self.learn_from_experience(observation, action, reward, next_observation, done, info['possible_actions'], step_nr,
                                 beta)

      observation = next_observation

      self.summary_writer.add_scalar('dqn/epsilon', epsilon, step_nr)

      if done:
        game_value = env.game_value()
        self.summary_writer.add_scalar('game_stats/end_turn', env.simulation.game.turn, step_nr)
        self.summary_writer.add_scalar('game_stats/game_value', (game_value + 1) / 2, step_nr)

        assert reward in (-1.0, 0.0, 1.0)
        observation, reward, terminal, info = env.reset()
      else:
        assert abs(reward) < 1

      if self.use_double_q and step_nr % target_update_every == 0:
        shared.sync_target(self.q_network, self.q_network_target)
      if step_nr % checkpoint_every == 0:
        torch.save(self.q_network.state_dict(), self.model_path)

  def choose(self, observation, info):
    board_center = observation.shape[1] // 2
    if self.should_flip_board:
      observation = np.concatenate(observation[board_center:], observation[:board_center], axis=1)

    action = self.q_network.act(observation, info['possible_actions'], 0.0)
    return action

  def act(self, state: np.array, possible_actions: List[Tuple[int, int]], epsilon: float, step_nr: int = None):
    assert isinstance(state, np.ndarray)
    assert isinstance(possible_actions, tuple)
    assert isinstance(possible_actions[0], tuple)
    assert isinstance(possible_actions[0][0], int)
    assert isinstance(epsilon, float)

    if random.random() > epsilon:
      q_values = self.get_q_values(self.q_network, state, possible_actions)
      # if step_nr is not None:
      #   self.summary_writer.add_histogram('q_values', q_values, global_step=step_nr)

      best_action = torch.argmax(q_values)
      action = possible_actions[best_action.detach()]
    else:
      action, = random.sample(possible_actions, 1)
    return action

  def __del__(self):
    self.summary_writer.close()


def train_gym_online_rl(gym_env, replay_buffer, model_type, trainer, predictor, test_run_name, score_bar, num_episodes,
                        max_steps, train_every_ts, train_after_ts, test_every_ts, test_after_ts, num_train_batches,
                        avg_over_num_episodes, render, save_timesteps_to_dataset, start_saving_from_episode, ):
  total_timesteps = 0
  avg_reward_history, timestep_history = [], []

  for i in range(num_episodes):
    terminal = False
    next_state = gym_env.env.reset()
    next_action = gym_env.policy(predictor, next_state, False)
    reward_sum = 0
    ep_timesteps = 0

    while not terminal:
      state = next_state
      action = next_action

      possible_actions, _ = get_possible_next_actions(gym_env, model_type, terminal)

      next_state, reward, terminal, _ = gym_env.env.step(action)

      ep_timesteps += 1
      total_timesteps += 1
      next_action = gym_env.policy(predictor, next_state, False)
      reward_sum += reward

      # Get possible next actions
      possible_next_actions, possible_next_actions_lengths = get_possible_next_actions(gym_env, model_type, terminal)

      replay_buffer.insert_into_memory(
        np.float32(state),
        action,
        np.float32(reward),
        np.float32(next_state),
        next_action,
        terminal,
        possible_next_actions,
        possible_next_actions_lengths,
        1,
      )

      if save_timesteps_to_dataset and i >= start_saving_from_episode:
        save_timesteps_to_dataset.insert(
          i,
          ep_timesteps - 1,
          state.tolist(),
          action_to_log,
          reward,
          terminal,
          possible_actions,
          1,
          1.0,
        )

      # Training loop
      if (
        total_timesteps % train_every_ts == 0
        and total_timesteps > train_after_ts
        and len(replay_buffer.replay_memory) >= trainer.minibatch_size
      ):
        for _ in range(num_train_batches):
          samples = replay_buffer.sample_memories(
            trainer.minibatch_size, model_type
          )
          samples.set_type(trainer.dtype)
          trainer.train(samples)

      # Evaluation loop
      if total_timesteps % test_every_ts == 0 and total_timesteps > test_after_ts:
        avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
          avg_over_num_episodes, predictor, test=True
        )
        avg_reward_history.append(avg_rewards)
        timestep_history.append(total_timesteps)
        logger.info(
          "Achieved an average reward score of {} over {} evaluations."
          " Total episodes: {}, total timesteps: {}.".format(
            avg_rewards, avg_over_num_episodes, i + 1, total_timesteps
          )
        )
        if score_bar is not None and avg_rewards > score_bar:
          logger.info(
            "Avg. reward history for {}: {}".format(
              test_run_name, avg_reward_history
            )
          )
          return avg_reward_history, timestep_history, trainer, predictor

      if max_steps and ep_timesteps >= max_steps:
        break

    # If the episode ended due to a terminal state being hit, log that
    if terminal and save_timesteps_to_dataset:
      save_timesteps_to_dataset.insert(
        i,
        ep_timesteps,
        next_state.tolist(),
        next_action_to_log,
        0.0,
        terminal,
        possible_next_actions,
        1,
        1.0,
      )

    # Always eval on last episode if previous eval loop didn't return.
    if i == num_episodes - 1:
      avg_rewards, avg_discounted_rewards = gym_env.run_ep_n_times(
        avg_over_num_episodes, predictor, test=True
      )
      avg_reward_history.append(avg_rewards)
      timestep_history.append(total_timesteps)
      logger.info(
        "Achieved an average reward score of {} over {} evaluations."
        " Total episodes: {}, total timesteps: {}.".format(
          avg_rewards, avg_over_num_episodes, i + 1, total_timesteps
        )
      )

  logger.info(
    "Avg. reward history for {}: {}".format(test_run_name, avg_reward_history)
  )
  return avg_reward_history, timestep_history, trainer, predictor


class DQNAgentHorizon(agents.base_agent.Agent):
  def __init__(self, num_inputs, num_actions) -> None:
    pass

  def train(self, env, game_steps=None, checkpoint_every=10000, target_update_every=100, ):
    num_episodes = 301
    max_steps = None
    train_every_ts = 100
    train_after_ts = 10
    test_every_ts = 100
    test_after_ts = 10
    num_train_batches = 1
    avg_over_num_episodes = 100
    render = False
    save_timesteps_to_dataset = None
    start_saving_from_episode = 0
    test_run_name = "test run"
    from ml.rl import OpenAIGymMemoryPool
    replay_buffer = OpenAIGymMemoryPool(params["max_replay_memory_size"])
    trainer = create_trainer(params["model_type"], params, rl_parameters, use_gpu, env)
    predictor = create_predictor(trainer, model_type, use_gpu)

    return train_gym_online_rl(
      env,
      replay_buffer,
      model_type,
      trainer,
      predictor,
      test_run_name,
      score_bar,
      num_episodes,
      max_steps,
      train_every_ts,
      train_after_ts, test_every_ts, test_after_ts, num_train_batches, avg_over_num_episodes, render,
      save_timesteps_to_dataset, start_saving_from_episode)

  def choose(self, observation, info):
    pass

  def __del__(self):
    self.summary_writer.close()


def get_possible_next_actions(gym_env, model_type, terminal):
  if model_type == ModelType.PYTORCH_DISCRETE_DQN.value:
    possible_next_actions = [
      0 if terminal else 1 for __ in range(gym_env.action_dim)
    ]
    possible_next_actions_lengths = gym_env.action_dim
  elif model_type == ModelType.PYTORCH_PARAMETRIC_DQN.value:
    if terminal:
      possible_next_actions = np.array([])
      possible_next_actions_lengths = 0
    else:
      possible_next_actions = np.eye(gym_env.action_dim)
      possible_next_actions_lengths = gym_env.action_dim
  elif model_type == ModelType.CONTINUOUS_ACTION.value:
    possible_next_actions = None
    possible_next_actions_lengths = 0
  elif model_type == ModelType.SOFT_ACTOR_CRITIC.value:
    possible_next_actions = None
    possible_next_actions_lengths = 0
  else:
    raise NotImplementedError()
  return possible_next_actions, possible_next_actions_lengths
