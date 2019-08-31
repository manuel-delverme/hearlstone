import argparse
import glob
import re

import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import game_utils
import hs_config
import shared.constants as C


def load_latest_checkpoint(checkpoint):
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

  # hs_config.tensorboard_dir = os.path.join(hs_config.log_dir,
  #                                          f"tensorboard/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}_{hs_config.comment}.pt")
  # if "DELETEME" in hs_config.tensorboard_dir:
  #   hs_config.tensorboard_dir = tempfile.mktemp()
  return hs_config.comment


def train(args):
  try:
    if args.p1:
      player = agents.learning.ppo_agent.PPOAgent(
          num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, experiment_id=None, device='cpu')
      game_manager = game_utils.GameManager(args.address, [args.p2 or 'heuristic', ])
      player.enjoy(game_manager, checkpoint_file=args.p1)

    else:
      if args.p2:
        raise ValueError
      latest_checkpoint = load_latest_checkpoint(args.load_checkpoint)
      player = agents.learning.ppo_agent.PPOAgent(
          num_inputs=C.STATE_SPACE, num_possible_actions=C.ACTION_SPACE, experiment_id=setup_logging())
      game_manager = game_utils.GameManager(args.address, [latest_checkpoint or 'random', ])
      player.self_play(game_manager, checkpoint_file=latest_checkpoint)

  except KeyboardInterrupt:
    print("Captured KeyboardInterrupt from user, quitting")
  finally:
    player.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--address", default="0.0.0.0:50052")
  parser.add_argument("--p1", default=None)
  parser.add_argument("--p2", default=None)
  parser.add_argument("--load_checkpoint", default=None)
  parser.add_argument("--comment", default=None)
  args = parser.parse_args()
  train(args)
