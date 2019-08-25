import functools
import os
import sys
from typing import Callable, Type

import torch

import agents.base_agent
import shared.constants as C

use_gpu = torch.cuda.is_available()
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
comment = "DELETEME" if DEBUG else ""
device = torch.device("cuda:0" if use_gpu else "cpu")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

print_every = 20
log_to_stdout = DEBUG


class Environment:
  ENV_DEBUG = False
  ENV_DEBUG_HEURISTIC = False
  ENV_DEBUG_METRICS = False
  single_process = DEBUG
  address = "0.0.0.0:50052"
  max_life = 30
  max_deck_size = 30
  max_mana = 10

  newest_opponent_prob = 0.5

  max_cards_in_board = 7
  max_cards_in_deck = 30
  max_entities_in_board = max_cards_in_board + 1

  max_cards_in_hand = 10
  connection = 'rpc'
  max_processes = 4 if connection == 'mmf' else 12

  max_actions_per_game = 100
  reward_type = C.RewardType.default

  @staticmethod
  def get_reward_shape(r, game):
    from shared.env_utils import get_extra_reward
    _r = get_extra_reward(game, reward_type=Environment.reward_type)
    return r + _r / Environment.max_actions_per_game

  @staticmethod
  def get_game_mode(address: str) -> Callable[[], Callable]:
    if Environment.connection == 'rpc':
      from environments.sabber_hs import Sabberstone as _Sabberstone
      print("Running as rpc")
    else:
      from environments.sabber2_hs import Sabberstone2 as _Sabberstone
      print("Running as mmf")

    out = functools.partial(
        _Sabberstone,
        address=address,
    )
    return out

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    return agents.heuristic.hand_coded.SabberAgent


class GameManager:
  arena = True
  num_battle_games = 5 if DEBUG else 50
  selection_size = 5
  league_size = 5 if DEBUG else 10
  elo_lr = 16
  base_rating = 1000
  elo_scale = torch.log(torch.Tensor([10.])) / 400
  tau = 1.  # temperature
  player_fname = os.path.join(log_dir, 'model', 'current_player.pt')
  model_paths = log_dir + '/model/*.pt_*'


class SelfPlay:
  num_opponent_updates = 9999999


class PPOAgent:
  BIG_NUMBER = 9999999999999
  performance_to_early_exit = 0.55
  performance_to_early_eval = 0.40
  num_outcomes_for_early_exit = 50
  min_iter_between_evals = 10

  num_eval_games = 10 if DEBUG else 1000
  clip_value_loss = True
  hidden_size = 256
  eval_interval = 50
  save_interval = 400
  save_dir = os.path.join(log_dir, "model")
  debug_dir = os.path.join(log_dir, "debug")

  actor_adam_lr = 7e-4
  critic_adam_lr = 1e-5

  num_processes = 1 if DEBUG else Environment.max_processes  # number of CPU processes
  if num_processes > 4 and Environment.connection == 'mmf':
    raise NotImplementedError(">4 processes seem to crash")

  num_steps = 256  # 32
  ppo_epoch = 6  # times ppo goes over the data

  num_env_steps = int(1e10)
  gamma = 0.99  # discount for rewards
  tau = 0.95  # gae parameter

  kl_coeff = 3.

  entropy_coeff = 1e-1  # 1e-1  # 0.043  # randomness, 1e-2 to 1e-4
  value_loss_coeff = 0.5

  max_grad_norm = 0.5  # any bigger gradient is clipped
  num_mini_batches = 5
  clip_epsilon = 0.2  # PPO paper

  num_updates = 50 if DEBUG else num_env_steps // num_steps // num_processes
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
