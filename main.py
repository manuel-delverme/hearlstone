import os
import sys

import torch

# import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
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

  player = agents.learning.ppo_agent.PPOAgent(
    num_inputs=dummy_hs_env.observation_space.shape[0],
    num_possible_actions=dummy_hs_env.action_space.n,
    log_dir=os.path.join(os.getcwd(), 'ppo_log'),
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
