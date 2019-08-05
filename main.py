import argparse
import glob
import re

import torch

import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import game_utils
import hs_config


def train(args):
  if not hs_config.comment and args.comment is None:
    import tkinter.simpledialog
    try:
      root = tkinter.Tk()
      hs_config.comment = tkinter.simpledialog.askstring("comment", "comment")
      root.destroy()
    except tkinter.TclError as _:
      print("no-comment")

  game_class = hs_config.Environment.get_game_mode(args.address)
  dummy_hs_env = game_class()
  num_actions = dummy_hs_env.action_space.n

  player = agents.learning.ppo_agent.PPOAgent(
      num_inputs=dummy_hs_env.observation_space.shape[0],
      num_possible_actions=num_actions,
  )
  del dummy_hs_env
  game_manager = game_utils.GameManager(address=args.address)

  if args.p1 is not None and args.p2 is None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    player.enjoy(game_manager, checkpoint_file=args.p1)

  elif args.p1 is not None and args.p2 is not None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    game_manager.add_learning_opponent(args.p1)
    player.enjoy(game_manager, checkpoint_file=args.p2)
  else:
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + f'*{hs_config.comment}*')
    # m = re.search('(?<=abc)def', 'abcdef')
    # m.group(0)
    checkpoint_files = sorted(checkpoints, key=lambda x: int(
        re.search(r"(?<=steps=)\w*(?=:)", x).group(0)
    ))
    if checkpoint_files:
      latest_checkpoint = checkpoint_files[-1]
    else:
      latest_checkpoint = None

    player.self_play(game_manager, checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--address", default="0.0.0.0:50052")
  parser.add_argument("--p1", default=None)
  parser.add_argument("--p2", default=None)
  parser.add_argument("--comment", default=None)
  args = parser.parse_args()
  train(args)
