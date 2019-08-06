import argparse
import glob
import re

import torch

import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import game_utils
import hs_config
import shared.constants as C


def load_latest_checkpoint():
  checkpoints = glob.glob(hs_config.PPOAgent.save_dir + f'*{hs_config.comment}*')
  checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=:)", x).group(0)))
  if checkpoint_files:
    checkpoint = checkpoint_files[-1]
  else:
    checkpoint = None
  return checkpoint


def train(args):
  if not hs_config.comment and args.comment is None:
    import tkinter.simpledialog
    try:
      root = tkinter.Tk()
      hs_config.comment = tkinter.simpledialog.askstring("comment", "comment")
      root.destroy()
    except tkinter.TclError as _:
      print("no-comment")

  player = agents.learning.ppo_agent.PPOAgent(num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, )
  game_manager = game_utils.GameManager(address=args.address)

  if args.p1 is not None and args.p2 is None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    player.enjoy(game_manager, checkpoint_file=args.p1)

  elif args.p1 is not None and args.p2 is not None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    game_manager.add_learned_opponent(args.p1)
    player.enjoy(game_manager, checkpoint_file=args.p2)
  else:
    latest_checkpoint = load_latest_checkpoint()
    player.self_play(game_manager, checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--address", default="0.0.0.0:50052")
  parser.add_argument("--p1", default=None)
  parser.add_argument("--p2", default=None)
  parser.add_argument("--comment", default=None)
  args = parser.parse_args()
  train(args)
