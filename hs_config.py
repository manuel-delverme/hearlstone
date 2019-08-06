import datetime
import functools
import os
import sys
import tempfile
from typing import Callable, Type

import torch

import agents.base_agent

use_gpu = False

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
comment = "DELETEME" if DEBUG else "d10c4n3"
device = torch.device("cuda:0" if use_gpu else "cpu")

print_every = 20
log_to_stdout = DEBUG

tensorboard_dir = os.path.join(f"logs/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{self.experiment_id}.pt")
if "DELETEME" in tensorboard_dir:
  tensorboard_dir = tempfile.mktemp()


class Environment:
  ENV_DEBUG = False
  ENV_DEBUG_HEURISTIC = False
  ENV_DEBUG_METRICS = False
  no_subprocess = True
  address = "0.0.0.0:50052"

  newest_opponent_prob = 0.5

  max_cards_in_board = 7
  max_entities_in_board = max_cards_in_board + 1

  max_cards_in_hand = 10

  max_turns = 50

  @staticmethod
  def get_game_mode(address: str) -> Callable[[], Callable]:
    import environments.sabber_hs
    out = functools.partial(
        environments.sabber_hs.Sabberstone,
        address
    )
    return out

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    return agents.heuristic.hand_coded.SabberAgent


class GameManager:
  max_opponents = 5
  old_opponent_prob = 0.2


class SelfPlay:
  num_opponent_updates = 99


class PPOAgent:
  BIG_NUMBER = 9999999999999
  winratio_cutoff = 0.8
  num_eval_games = 10 if DEBUG else 100
  clip_value_loss = True
  hidden_size = 256  # 64
  eval_interval = 40
  save_interval = 100
  _log_dir = os.path.join(os.path.dirname(os.getcwd()), "hearlstone", "logs")
  save_dir = os.path.join(_log_dir, "model")
  debug_dir = os.path.join(_log_dir, "debug")

  adam_lr = 7e-4

  num_processes = 2 if DEBUG else 6  # number of CPU processes
  num_steps = 32
  ppo_epoch = 4  # times ppo goes over the data

  num_env_steps = int(3e4)
  gamma = 0.99  # discount for rewards
  tau = 0.95  # gae parameter

  entropy_coeff = 1e-1  # 0.043  # randomness, 1e-2 to 1e-4
  value_loss_coeff = 0.5
  max_grad_norm = 0.5  # any bigger gradient is clipped
  num_mini_batches = 5
  clip_epsilon = 0.2  # PPO paper

  num_updates = 2 if DEBUG else num_env_steps // num_steps // num_processes


if any((Environment.ENV_DEBUG, Environment.ENV_DEBUG, Environment.ENV_DEBUG_HEURISTIC, Environment.ENV_DEBUG_METRICS,
        Environment.no_subprocess)):
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
