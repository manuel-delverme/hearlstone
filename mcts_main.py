import torch

import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import hs_config


def make_env(seed=None, env_id=None, log_dir=None, episode_life=None):
  hs_game = hs_config.VanillaHS.get_game_mode()()
  hs_game.set_opponent(opponent=hs_config.VanillaHS.get_opponent()())
  return hs_game


def train():
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
    num_inputs=dummy_hs_env.observation_space.shape,
    action_space=dummy_hs_env.action_space,
    log_dir='/tmp/ppo_log/',
    record=not hs_config.enjoy,
  )
  del dummy_hs_env

  if hs_config.enjoy:
    player.load_model()
    player.render(make_env())
  else:
    player.train(make_env, hs_config.seed)


if __name__ == "__main__":
  train()
