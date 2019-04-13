import glob

import torch

# import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import game_utils
import hs_config


def train():
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
    num_inputs=dummy_hs_env.observation_space.shape[0],
    num_possible_actions=dummy_hs_env.action_space.n,
    log_dir='/tmp/ppo_log/',
  )
  del dummy_hs_env
  game_manager = game_utils.GameManager(hs_config.seed)

  if hs_config.enjoy:
    checkpoints = glob.glob(hs_config.PPOAgent.save_dir + '*Vanilla*')
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.replace(":", "-").split("-")[-1][:-3]))[-1]
    # checkpoints = glob.glob(hs_config.PPOAgent.save_dir + '*Vanilla*{}-*'.format(hs_config.VanillaHS.level))
    player.enjoy(game_manager, checkpoint_file=latest_checkpoint)
  else:
    player.self_play(game_manager, checkpoint_file=None)  # , checkpoint_file=latest_checkpoint)


if __name__ == "__main__":
  train()
