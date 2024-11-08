import argparse
import glob
import re

import torch

import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import game_utils
import hs_config


def load_latest_checkpoint(checkpoint=None):
  if checkpoint is None:
    print('loading checkpoints', hs_config.PPOAgent.save_dir + f'/*{hs_config.comment}*')
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + f'/*{hs_config.comment}*')
    if checkpoints:
      # checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=:)", x).group(0)))
      checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=\.pt)", x).group(0)))
      checkpoint = checkpoint_files[-1]
  print('found', checkpoint)
  return checkpoint


def setup_logging():
  if args.comment is not None:
    hs_config.comment = args.comment
  elif not hs_config.comment:
    import tkinter.simpledialog
    try:
      root = tkinter.Tk()
      hs_config.comment = tkinter.simpledialog.askstring("comment", "comment")
      root.destroy()
    except tkinter.TclError as _:
      print("no-comment")
  return f"{str(hs_config.Environment.game_mode).split('.')[-1]}:{hs_config.comment}"


def train(args):
  game_manager = game_utils.GameManager()
  if args.p1 is not None and args.p2 is None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=game_manager.observation_space.shape[0],
                                                num_possible_actions=game_manager.action_space.n,
                                                experiment_id=None)

    player.enjoy(game_manager, checkpoint_file=args.p1)

  elif args.p1 is not None and args.p2 is not None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    game_manager.add_learned_opponent(args.p2)
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=game_manager.observation_space.shape[0],
                                                num_possible_actions=game_manager.action_space.n,
                                                experiment_id=None)
    player.enjoy(game_manager, checkpoint_file=args.p1)
  else:
    experiment_id = setup_logging()
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=game_manager.observation_space.shape[0],
                                                num_possible_actions=game_manager.action_space.n,
                                                experiment_id=experiment_id)
    latest_checkpoint = load_latest_checkpoint(args.load_checkpoint)
    player.self_play(game_manager, checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--address", default="0.0.0.0:50052")
  parser.add_argument("--p1", default=None)
  parser.add_argument("--p2", default=None)
  parser.add_argument("--load_checkpoint", default=None)
  parser.add_argument("--comment", default=None)
  args = parser.parse_args()
  train(args)
