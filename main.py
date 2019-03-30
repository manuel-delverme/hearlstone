import glob
import sys

import torch

import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import hs_config


def make_env(seed=None, env_id=None, log_dir=None, episode_life=None):
  hs_game = hs_config.VanillaHS.get_game_mode()(seed=seed)
  hs_game.set_opponent(opponent=hs_config.VanillaHS.get_opponent()())
  return hs_game


def train():
  if '_pydev_bundle.pydev_log' in sys.modules.keys():
    hs_config.PPOAgent.num_processes = 2
    hs_config.comment = "DELETEME"

  if not hs_config.comment:
    import tkinter.simpledialog
    # comment = "256h32bs"
    root = tkinter.Tk()
    hs_config.comment = tkinter.simpledialog.askstring("comment", "comment")
    root.destroy()

  if hs_config.seed:
    torch.manual_seed(hs_config.seed)
    torch.cuda.manual_seed_all(hs_config.seed)

    if hs_config.use_gpu and torch.cuda.is_available() and hs_config.make_deterministic:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True

  agent = agents.learning.ppo_agent.PPOAgent
  game_class = hs_config.VanillaHS.get_game_mode()

  dummy_hs_env = game_class()
  player = agent(
    observation_space=dummy_hs_env.observation_space,
    action_space=dummy_hs_env.action_space,
    log_dir='/tmp/ppo_log/',
  )
  del dummy_hs_env

  if hs_config.enjoy:
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + '*Vanilla*')
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.replace(":", "-").split("-")[-1][:-3]))[-1]
    # checkpoints = glob.glob(hs_config.PPOAgent.save_dir + '*Vanilla*{}-*'.format(hs_config.VanillaHS.level))
    player.enjoy(make_env, checkpoint_file=latest_checkpoint)
  else:
    player.train(make_env, hs_config.seed, checkpoint_file=None)  # , checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  train()
