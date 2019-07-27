import sys
from typing import Callable, Type

import torch

# DO NOT ADD PROJECT LEVEL IMPORTS OR CYCLES!
import agents.base_agent
from environments import base_env

enjoy = False
# game_mode = "trading"
use_gpu = False
seed = 1337
benchmark = False
make_deterministic = False  # Supposedly slows by a lot

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

comment = "DELETEME" if DEBUG else ""  # d10c4n3
device = torch.device("cuda:0" if use_gpu else "cpu")
# device = 'gpu' if use_gpu else 'cpu'

print_every = 20
BIG_NUMBER = 9999999999999
visualize_everything = 0


class Environment:
  render_after_step = visualize_everything or 0
  _DEBUG = DEBUG
  no_subprocess = _DEBUG or 0
  old_opponent_prob = 0.2
  sort_decks = False
  normalize = True
  starting_hp = 30

  # this is now level
  max_cards_in_board = 7
  max_cards_in_hand = 10
  max_turns = 50

  always_first_player = True

  level = 4

  @staticmethod
  def get_game_mode() -> Callable[[], base_env.BaseEnv]:
    # if game_mode == "trading":
    # import environments.tutorial_environments
    # return environments.tutorial_environments.TradingHS
    # else:
    # import environments.vanilla_hs
    # return environments.vanilla_hs.VanillaHS
    # import sb_env.SabberStone_python_client.simulator
    # return sb_env.SabberStone_python_client.simulator.Sabbertsone
    import environments.sabber_hs
    return environments.sabber_hs.Sabbertsone

  @staticmethod
  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    # import agents.heuristic.random_agent
    # return agents.heuristic.hand_coded.PassingAgent
    # return agents.heuristic.hand_coded.TradingAgent
    # return agents.heuristic.random_agent.RandomAgent
    return agents.heuristic.hand_coded.SabberAgent
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

  num_processes = 2 if DEBUG else 6  # number of CPU processes
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
  load_experts = False


if Environment._DEBUG:
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
