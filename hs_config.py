import sys
from typing import Callable, Type

import torch

# DO NOT ADD PROJECT LEVEL IMPORTS OR CYCLES!
import agents.base_agent
from environments import base_env

enjoy = False
use_gpu = True
seed = 1337
benchmark = False
make_deterministic = False  # Supposedly slows by a lot

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

comment = "DELETEME" if DEBUG else ""
device = torch.device("cuda:0" if use_gpu else "cpu")
# device = 'gpu' if use_gpu else 'cpu'

print_every = 20
BIG_NUMBER = 9999999999999


class VanillaHS:
  DEBUG = False
  no_subprocess = False
  old_opponent_prob = 0.2
  sort_decks = False
  normalize = True
  starting_hp = 30

  # this is now level
  max_cards_in_board = 7
  max_cards_in_hand = 10

  always_first_player = False

  level = 6

  @staticmethod
  def get_game_mode() -> Callable[[], base_env.BaseEnv]:
    # import environments.tutorial_environments
    # return environments.tutorial_environments.TradingHS
    import environments.vanilla_hs
    return environments.vanilla_hs.VanillaHS

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    # return agents.heuristic.hand_coded.PassingAgent
    return agents.heuristic.hand_coded.HeuristicAgent
    # hs_config.VanillaHS.get_opponent = get_opponent
    # return agents.learning.ppo_agent.PPOAgent


class SelfPlay:
  num_opponent_updates = 99


class PPOAgent:
  # Monitoring
  winratio_cutoff = 0.8
  num_eval_games = 10 if DEBUG else 100
  clip_value_loss = True
  hidden_size = 256  # 64
  eval_interval = 40
  save_interval = 100
  save_dir = "ppo_save_dir/"
  # Optimizer
  adam_lr = 7e-4
  # adam_lr = 2.5e-4  # 7e-4 in reference implementation

  # Algorithm use_linear_clip_decay = False
  use_linear_lr_decay = False

  num_processes = 2 if DEBUG else 8  # number of CPU processes
  num_steps = 32
  ppo_epoch = 4  # times ppo goes over the data

  num_env_steps = int(3e4)
  gamma = 0.99  # discount for rewards
  tau = 0.95  # gae parameter

  entropy_coeff = 0.043  # randomness, 1e-2 to 1e-4
  value_loss_coeff = 0.5
  max_grad_norm = 0.5  # any bigger gradient is clipped
  num_mini_batches = 5
  clip_epsilon = 0.2  # PPO paper

  num_updates = 2 if DEBUG else num_env_steps // num_steps // num_processes
