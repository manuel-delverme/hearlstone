from typing import Callable, Type
import gym

import torch
import torch.optim as optim

# DO NOT ADD PROJECT LEVEL IMPORTS OR CYCLES!
import agents.base_agent

enjoy = True
use_gpu = False
seed = 1337
benchmark = False
make_deterministic = False # Supposedly slows by a lot

device = torch.device("cuda:0" if use_gpu else "cpu")
# device = 'gpu' if use_gpu else 'cpu'

log_interval = 10
BIG_NUMBER = 9999999999999



class VanillaHS:
  sort_decks = False
  debug = False
  normalize = True
  starting_hp = 30
  max_cards_in_board = 5
  max_cards_in_hand = 5
  always_first_player = True

  @staticmethod
  def get_game_mode() -> Callable[[], gym.Env]:
    import environments.tutorial_environments
    return environments.tutorial_environments.TradingHS

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    return agents.heuristic.hand_coded.PassingAgent


class DQNAgent:
  nr_parallel_envs = 1
  nr_epochs = nr_parallel_envs
  tau = 1
  gradient_clip = 1
  silly = False
  target_update = 1000
  buffer_size = int(1e6)
  training_steps = int(1e7)
  # warmup_steps = int(5e3)
  warmup_steps = 0

  epsilon_decay = None
  beta_decay = max(training_steps / 3, int(1e5))
  # optimizer = optim.RMSprop

  optimizer = optim.Adam
  gamma = 0.99
  # lr = 1e-5
  lr = 1e-3
  l2_decay = 0
  # batch_size = 256
  batch_size = 32


class PPOAgent:
  # Monitoring
  eval_interval = 10
  save_interval = 100
  save_dir = "ppo_save_dir/"
  add_timestep = False  # Adds the time step to observations

  # Optimizer
  adam_lr = 2.5e-4  # 7e-4 in reference implementation

  # Algorithm
  use_gae = True
  use_linear_clip_decay = False
  use_linear_lr_decay = False

  num_processes = 1  # 6  # number of CPU processes
  num_steps = 20
  ppo_epoch = 4  # times ppo goes over the data

  num_env_steps = 15000
  gamma = 0.99  # discount for rewards
  tau = 0.95  # gae parameter

  entropy_coeff = 0.01  # entropy weight in loss function
  value_loss_coeff = 0.5
  max_grad_norm = 0.5  # any bigger gradient is clipped
  num_mini_batches = 32 if num_processes > 1 else num_steps
  clip_epsilon = 0.2  # PPO paper

  # batch_size = 5000
  use_joint_pol_val = True

  num_updates = int(num_env_steps) // num_steps // num_processes


class A2CAgent:
  num_workers = 64
  optimizer = optim.Adam
  lr = 1e-5
  discount = 0.99
  use_gae = True
  gae_tau = 0.95
  checkpoint_every = int(2e5)
  entropy_weight = 0.001
  rollout_length = 5
  gradient_clip = 0.5
  # defaults
  training_steps = int(1e7)

  # Hyper params:
  hidden_size = 32


