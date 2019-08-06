# import baselines.common.running_mean_std
import collections

import torch

import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.models.randomized_policy
import hs_config


class GameManager(object):
  def __init__(self, address=hs_config.Environment.address):
    self.game_class = hs_config.Environment.get_game_mode(address)
    self.opponents = collections.deque([agents.heuristic.random_agent.RandomAgent()], maxlen=hs_config.GameManager.max_opponents)
    self.game_matrix = []

  def __call__(self):
    hs_game = self.game_class()
    if self.use_heuristic_opponent:
      hs_game.set_opponents(opponents=[hs_config.Environment.get_opponent()()])
    else:
      hs_game.set_opponents(opponents=self.opponents, opponent_obs_rmss=self.opponent_normalization_factors)

    return hs_game

  def add_learned_opponent(self, checkpoint_file):
    import agents.learning.ppo_agent

    if self.use_heuristic_opponent:
      assert isinstance(self.opponents[0], agents.heuristic.random_agent.RandomAgent)
      self.opponents = []
      self.opponent_normalization_factors = []
      self.use_heuristic_opponent = False
    opponent_network, = torch.load(checkpoint_file)

    assert isinstance(opponent_network, agents.learning.models.randomized_policy.ActorCritic), opponent_network
    opponent = agents.learning.ppo_agent.PPOAgent(opponent_network.num_inputs, opponent_network.num_possible_actions)

    del opponent.pi_optimizer
    del opponent.value_optimizer
    opponent_network.eval()
    for network in (opponent_network.actor, opponent_network.critic, opponent_network.actor_logits):
      for param in network.parameters():
        param.requires_gradient = False

    opponent.actor_critic = opponent_network
    if hasattr(opponent, 'logger'):  # the logger has rlocks which cannot change process so RIP
      del opponent.logger  # TODO: this is an HACK, refactor it away
    if hasattr(opponent, 'timer'):  # the timer has rlocks which cannot change process so RIP
      del opponent.timer  # TODO: this is an HACK, refactor it away

    self.opponents.append(opponent)
