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


def load_latest_checkpoint(checkpoint=None):
  if checkpoint is None:
    print('loading checkpoints', hs_config.PPOAgent.save_dir + f'/*{hs_config.comment}*')
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + f'/*{hs_config.comment}*')
    if checkpoints:
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

  return hs_config.comment


def train(args):
  game_manager = game_utils.GameManager(address=args.address)
  game_manager.reset()

  if args.p1 is not None and args.p2 is None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, experiment_id=None)

    player.enjoy(game_manager, checkpoint_file=args.p1)

  elif args.p1 is not None and args.p2 is not None:
    hs_config.use_gpu, hs_config.device = False, torch.device('cpu')
    game_manager.add_learning_opponent(args.p1)
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, experiment_id=None)
    player.enjoy(game_manager, checkpoint_file=args.p2)
  else:
    experiment_id = setup_logging()
    latest_checkpoint = load_latest_checkpoint(args.load_checkpoint)
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, experiment_id=experiment_id)
    init_ckpt = player.save_model(0)

    if hs_config.GameManager.arena:
      game_manager.create_league(init_ckpt)
      for idx, ckpt in enumerate(game_manager.model_list):
        game_manager.elo.set_player_index(idx)
        rewards, scores = player.battle(game_manager, checkpoint_file=ckpt)
        game_manager.update_score(scores)
      del player.battle_env
      game_manager.set_selection()
    else:
      game_manager.add_learned_opponent(init_ckpt)
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
