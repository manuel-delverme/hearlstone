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


def load_latest_checkpoint(checkpoint=None, experiment_id=hs_config.comment):
  if checkpoint is None:
    print('loading checkpoints', hs_config.PPOAgent.save_dir + f'/*{experiment_id}*')
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + f'/*{experiment_id}*')
    if checkpoints:
      # specs = list(map(get_ckpt_specs, checkpoints))
      # checkpoint_files = sorted(zip(checkpoints, specs), key=lambda x: (int(x[1].score), int(x[1].steps)))
      # checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=:)", x).group(0)))
      checkpoint_files = sorted(checkpoints, key=lambda x: int(re.search(r"(?<=steps=)\w*(?=\.pt)", x).group(0)))
      checkpoint = checkpoint_files[-1]
  print('found', checkpoint)
  return checkpoint


def get_ckpt_specs(file_name):
  # TODO make me a regex
  import collections
  import os
  FileName = collections.namedtuple('FileName', ['id', 'steps', 'score'], defaults=(None, None, 0))
  file_name = file_name.split('/')[-1]
  file_name = os.path.splitext(file_name)[0]
  ret = file_name.split(':')
  ret = [r.split('=')[1] for r in ret]
  file_name = FileName(*ret)
  return file_name


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

  # hs_config.tensorboard_dir = os.path.join(hs_config.log_dir,
  #                                          f"tensorboard/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{hs_config.comment}.pt")
  # if "DELETEME" in hs_config.tensorboard_dir:
  #   hs_config.tensorboard_dir = tempfile.mktemp()
  return hs_config.comment


def train(args):
  game_manager = game_utils.GameManager(address=args.address)

  if hs_config.Environment.arena is True:
    for experiment_id in hs_config.Environment.opponent_keys:
      ckpt = load_latest_checkpoint(experiment_id=experiment_id)
      if ckpt is not None:
        print(f"[ARENA] Loading ckpt {ckpt}")
        game_manager.add_learned_opponent(ckpt)

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
    player = agents.learning.ppo_agent.PPOAgent(num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, experiment_id=experiment_id)
    latest_checkpoint = None
    if args.load_latest:
      latest_checkpoint = load_latest_checkpoint(experiment_id=experiment_id)
      print(f"[MAIN] Found latest checkpoint, {latest_checkpoint}")
    player.self_play(game_manager, checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--address", default="0.0.0.0:50052")
  parser.add_argument("--p1", default=None)
  parser.add_argument("--p2", default=None)
  parser.add_argument("--load-latest", default=True)
  parser.add_argument("--comment", default=None)
  args = parser.parse_args()
  train(args)
