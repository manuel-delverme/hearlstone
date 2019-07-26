import os
import sys

import torch

# import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import environments.tutorial_environments
import game_utils
import hs_config


# torch.cuda.current_device()  # this is required for M$win driver to work


def train():
  if not hs_config.comment and len(sys.argv) == 1:
    import tkinter.simpledialog
    # comment = "256h32bs"
    try:
      root = tkinter.Tk()
      hs_config.comment = tkinter.simpledialog.askstring("comment", "comment")
      root.destroy()
    except tkinter.TclError as _:
      print("no-comment")

  if hs_config.seed:
    torch.manual_seed(hs_config.seed)
    torch.cuda.manual_seed_all(hs_config.seed)

    if hs_config.use_gpu and torch.cuda.is_available() and hs_config.make_deterministic:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True

  game_class = hs_config.Environment.get_game_mode()
  dummy_hs_env = game_class()
  num_actions = dummy_hs_env.action_space.n

  if not hs_config.PPOAgent.load_experts or (
    hs_config.Environment.get_game_mode() is environments.tutorial_environments.TradingHS):
    experts = tuple()
  else:
    trading_expert = agents.learning.ppo_agent.Expert(
      "/home/esac/projects/hearlstone/ppo_save_dir/id=TradingHS-4-d10c4n3:steps=96:inputs=110.pt")
    experts = (trading_expert,)
    num_actions += len(experts)

  player = agents.learning.ppo_agent.PPOAgent(
    num_inputs=dummy_hs_env.observation_space.shape[0],
    num_possible_actions=num_actions,
    log_dir=os.path.join(os.getcwd(), 'ppo_log'),
    experts=experts,
  )
  del dummy_hs_env
  game_manager = game_utils.GameManager(hs_config.seed)

  if len(sys.argv) == 2:
    hs_config.use_gpu = False
    hs_config.device = torch.device('cpu')
    # checkpoints = glob.glob(hs_config.PPOAgent.save_dir + '*Vanilla*')
    # latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.replace(":", "-").split("-")[-1][:-3]))[-1]

    # checkpoints = glob.glob(hs_config.PPOAgent.save_dir + '*Vanilla*{}-*'.format(hs_config.VanillaHS.level))

    player.enjoy(game_manager, checkpoint_file=sys.argv[1])
  elif len(sys.argv) == 3:
    hs_config.use_gpu = False
    hs_config.device = torch.device('cpu')
    game_manager.add_learning_opponent(sys.argv[2])
    player.enjoy(game_manager, checkpoint_file=sys.argv[1])
  else:
    # pass
    # player.self_play(game_manager, checkpoint_file=None)  # , checkpoint_file=latest_checkpoint)
    player.train(game_manager, checkpoint_file=None)  # , checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  train()
