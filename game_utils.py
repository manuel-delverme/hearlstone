import tempfile

import torch

import agents.base_agent
import agents.heuristic.random_agent
import agents.learning.ppo_agent
import hs_config


class GameManager(object):
  def __init__(self, seed=None, env_id=None, log_dir=None):
    self.seed = seed
    self.game_class = hs_config.VanillaHS.get_game_mode()

    self.opponent = agents.heuristic.random_agent.RandomAgent()
    self.opponent_obs_rms = None

  def __call__(self, extra_seed):
    assert isinstance(self.opponent, (agents.base_agent.Bot, agents.base_agent.Agent))
    hs_game = self.game_class(seed=self.seed, extra_seed=extra_seed)
    hs_game.set_opponent(opponent=self.opponent, opponent_obs_rms=self.opponent_obs_rms)
    return hs_game

  def update_learning_opponent(self, checkpoint_file):
    opponent_network, self.opponent_obs_rms = torch.load(checkpoint_file)

    # return agents.learning.ppo_agent.PPOAgent
    self.opponent = agents.learning.ppo_agent.PPOAgent(
      opponent_network.num_inputs, opponent_network.num_possible_actions, log_dir=tempfile.mktemp())

    del self.opponent.optimizer
    opponent_network.eval()
    for network in (opponent_network.actor, opponent_network.critic, opponent_network.actor_logits):
      for param in network.parameters():
        param.requires_gradient = False

    self.opponent.actor_critic = opponent_network

  def set_heuristic_opponent(self):
    self.opponent = hs_config.VanillaHS.get_opponent()()
    self.opponent_obs_rms = None
