# import baselines.common.running_mean_std
import collections
import tempfile

import torch

import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.models.randomized_policy
import agents.learning.ppo_agent
import hs_config
from baselines_repo.baselines.common.running_mean_std import RunningMeanStd


class GameManager(object):
  def __init__(self, seed=None, env_id=None, log_dir=None, address=None):
    self.seed = seed
    self._use_heuristic_opponent = True

    self.game_class = hs_config.Environment.get_game_mode(address)
    self.opponents = collections.deque([agents.heuristic.random_agent.RandomAgent()],
                                       maxlen=hs_config.Environment.max_opponents)
    self.opponent_normalization_factors = [None]
    self.game_matrix = []

  @property
  def use_heuristic_opponent(self):
    return self._use_heuristic_opponent

  @use_heuristic_opponent.setter
  def use_heuristic_opponent(self, value):
    self._use_heuristic_opponent = value

  def __call__(self, extra_seed):
    hs_game = self.game_class(seed=self.seed, extra_seed=extra_seed)
    if self.use_heuristic_opponent:
      hs_game.set_opponents(opponents=[hs_config.Environment.get_opponent()()], opponent_obs_rmss=[None, ])
    else:
      hs_game.set_opponents(opponents=self.opponents, opponent_obs_rmss=self.opponent_normalization_factors)

    return hs_game

  def add_learning_opponent(self, checkpoint_file):
    if self.use_heuristic_opponent:
      assert isinstance(self.opponents[0], agents.heuristic.random_agent.RandomAgent)
      self.opponents = []
      self.opponent_normalization_factors = []
      self.use_heuristic_opponent = False
    opponent_network, opponent_obs_rms = torch.load(checkpoint_file)

    assert isinstance(opponent_network, agents.learning.models.randomized_policy.ActorCritic), opponent_network
    assert (opponent_obs_rms is None or isinstance(opponent_obs_rms, RunningMeanStd)), opponent_obs_rms

    opponent = agents.learning.ppo_agent.PPOAgent(opponent_network.num_inputs, opponent_network.num_possible_actions,
                                                  log_dir=tempfile.mktemp())

    del opponent.logger
    del opponent.optimizer
    opponent_network.eval()
    for network in (opponent_network.actor, opponent_network.critic, opponent_network.actor_logits):
      for param in network.parameters():
        param.requires_gradient = False

    opponent.actor_critic = opponent_network

    self.opponents.append(opponent)
    self.opponent_normalization_factors.append(opponent_obs_rms)

