import functools
import os
import sys
from typing import Callable, Type

import torch

import agents.base_agent

use_gpu = torch.cuda.is_available()
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
comment = "DELETEME" if DEBUG else ""
device = torch.device("cuda:0" if use_gpu else "cpu")

print_every = 20
log_to_stdout = DEBUG


class Environment:
  ENV_DEBUG = False
  ENV_DEBUG_HEURISTIC = False
  ENV_DEBUG_METRICS = False
  single_process = DEBUG
  address = "0.0.0.0:50052"

  newest_opponent_prob = 0.8

  max_cards_in_board = 7
  max_cards_in_deck = 30
  max_entities_in_board = max_cards_in_board + 1

  max_cards_in_hand = 10

  @staticmethod
  def get_game_mode(address: str) -> Callable[[], Callable]:
    import environments.sabber2_hs
    out = functools.partial(
        environments.sabber2_hs.Sabberstone2,
        address=address,
    )
    return out

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    return agents.heuristic.hand_coded.SabberAgent


class GameManager:
  max_opponents = 5
  elo_lr = 16
  base_rating = 1000
  elo_scale = torch.log(torch.Tensor([10.])) / 400


class SelfPlay:
  num_opponent_updates = 99


log_dir = os.path.join(os.path.dirname(os.getcwd()), "hearlstone", "logs")


class PPOAgent:
  BIG_NUMBER = 9999999999999
  performance_to_early_exit = 0.55
  num_episodes_for_early_exit = 50
  min_iter_between_evals = 10

  num_eval_games = 10 if DEBUG else 100
  clip_value_loss = True
  hidden_size = 256
  eval_interval = 50
  save_interval = 400
  save_dir = os.path.join(log_dir, "model")
  debug_dir = os.path.join(log_dir, "debug")

  actor_adam_lr = 7e-4
  critic_adam_lr = 1e-5

  num_processes = 1 if DEBUG else 4  # number of CPU processes
  if num_processes > 4:
    raise NotImplementedError(">4 processes seem to crash")
  num_steps = 32
  ppo_epoch = 6  # times ppo goes over the data

  num_env_steps = int(1e10)
  gamma = 0.99  # discount for rewards
  tau = 0.95  # gae parameter

  entropy_coeff = 1e-1  # 0.043  # randomness, 1e-2 to 1e-4
  value_loss_coeff = 0.5

  max_grad_norm = 0.5  # any bigger gradient is clipped
  num_mini_batches = 5
  clip_epsilon = 0.2  # PPO paper

  num_updates = 2 if DEBUG else num_env_steps // num_steps // num_processes
  assert num_updates


if any((DEBUG, Environment.ENV_DEBUG, Environment.ENV_DEBUG_HEURISTIC, Environment.ENV_DEBUG_METRICS,
        Environment.single_process)):
  print('''
                                    _.---"'"""""'`--.._
                             _,.-'                   `-._
                         _,."                            -.
                     .-""   ___...---------.._             `.
                     `---'""                  `-.            `.
                                                 `.            \
                                                   `.           \
                                                     \           \
                                                      .           \
                                                      |            .
                                                      |            |
                                _________             |            |
                          _,.-'"         `"'-.._      :            |
                      _,-'                      `-._.'             |
                   _.'                              `.             '
        _.-.    _,+......__                           `.          .
      .'    `-"'           `"-.,-""--._                 \        /
     /    ,'                  |    __  \                 \      /
    `   ..                       +"  )  \                 \    /
     `.'  \          ,-"`-..    |       |                  \  /
      / " |        .'       \   '.    _.'                   .'
     |,.."--"""--..|    "    |    `""`.                     |
   ,"               `-._     |        |                     |
 .'                     `-._+         |                     |
/                           `.                        /     |
|    `     '                  |                      /      |
`-.....--.__                  |              |      /       |
   `./ "| / `-.........--.-   '              |    ,'        '
     /| ||        `.'  ,'   .'               |_,-+         /
    / ' '.`.        _,'   ,'     `.          |   '   _,.. /
   /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
  /... _.`:.________,.'              `._,.-..|        "'
 `.__.'                                 `._  /
                                           "' mh
  ''')
