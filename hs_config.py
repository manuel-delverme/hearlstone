from typing import Callable, Type

import gym
import torch

# DO NOT ADD PROJECT LEVEL IMPORTS OR CYCLES!
import agents.base_agent

enjoy = False
use_gpu = True
seed = 1337
benchmark = False
make_deterministic = False  # Supposedly slows by a lot

comment = ""
device = torch.device("cuda:0" if use_gpu else "cpu")
# device = 'gpu' if use_gpu else 'cpu'

print_every = 20
BIG_NUMBER = 9999999999999


class VanillaHS:
  sort_decks = False
  debug = False
  normalize = True
  starting_hp = 30

  # this is now level
  max_cards_in_board = 7
  max_cards_in_hand = 10

  always_first_player = True

  level = 5

  @staticmethod
  def get_game_mode() -> Callable[[], gym.Env]:
    # import environments.tutorial_environments
    # return environments.tutorial_environments.TradingHS
    import environments.vanilla_hs
    return environments.vanilla_hs.VanillaHS

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Bot]:
    import agents.heuristic.hand_coded
    # return agents.heuristic.hand_coded.PassingAgent
    return agents.heuristic.hand_coded.HeuristicAgent


class PPOAgent:
  # Monitoring
  num_eval_games = 100
  clip_value_loss = True
  hidden_size = 256  # 64
  eval_interval = print_every * 5
  save_interval = 100
  save_dir = "ppo_save_dir/"
  # Optimizer
  adam_lr = 2.5e-4  # 7e-4 in reference implementation

  # Algorithm use_linear_clip_decay = False
  use_linear_lr_decay = False

  num_processes = 8  # number of CPU processes
  num_steps = 32
  ppo_epoch = 4  # times ppo goes over the data

  num_env_steps = 100000
  gamma = 0.99  # discount for rewards
  tau = 0.95  # gae parameter

  entropy_coeff = 1e-3  # randomness, 1e-2 to 1e-4
  value_loss_coeff = 0.5
  max_grad_norm = 0.5  # any bigger gradient is clipped
  num_mini_batches = 5
  clip_epsilon = 0.2  # PPO paper

  num_updates = int(num_env_steps) // num_steps // num_processes
