# import baselines.common.running_mean_std
import collections

import torch

import agents.base_agent
import agents.heuristic.hand_coded
import agents.heuristic.random_agent
import agents.learning.models.randomized_policy
import hs_config


class GameManager(object):
  def __init__(self, seed=None, env_id=None, log_dir=None, address=hs_config.Environment.address):
    assert log_dir is None
    assert env_id is None
    self.seed = seed
    self._use_heuristic_opponent = True

    self.game_class = hs_config.Environment.get_game_mode(address)
    self.opponents = collections.deque([agents.heuristic.random_agent.RandomAgent()], maxlen=hs_config.GameManager.max_opponents)
    self.opponent_normalization_factors = [None]

  @property
  def use_heuristic_opponent(self):
    return self._use_heuristic_opponent

  @use_heuristic_opponent.setter
  def use_heuristic_opponent(self, value):
    self._use_heuristic_opponent = value

  def __call__(self):
    hs_game = self.game_class()
    if self.use_heuristic_opponent:
      hs_game.set_opponents(opponents=[hs_config.Environment.get_opponent()()], opponent_obs_rmss=[None, ])
    else:
      hs_game.set_opponents(opponents=self.opponents, opponent_obs_rmss=self.opponent_normalization_factors)

    return hs_game

  def add_learning_opponent(self, checkpoint_file, score=None):
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
    self.scores.append(score)  # TODO: score may become a torch variable at some point


class Elo:
  def __init__(self):
    # https://arxiv.org/pdf/1806.02643.pdf

    max_opponents = hs_config.GameManager.max_opponents

    self.scores = torch.ones((max_opponents + 1,)) * hs_config.GameManager.elo_lr
    self.c = torch.rand(size=(max_opponents + 1, 2))
    self.alpha = hs_config.GameManager.elo_scale
    self.k = hs_config.GameManager.elo_lr
    self._player_idx = -1  # last on the list

  def __getitem__(self, item: int) -> float:
    return self.scores[item]

  def update(self, scores: dict):
    for idx, score in scores.items():
      idx += 1  # off by one from game manager distribution
      p = torch.Tensor(score).clamp(0, 1).mean()
      p_hat = self.__call__(idx)
      delta = p - p_hat
      self._grad(delta, opponent_idx=idx)

  def _grad(self, delta: float, opponent_idx: int):
    r_update = (self.k * delta, - self.k * delta)
    c_update = [[delta * self.c[opponent_idx, 1], - delta * self.c[self._player_idx, 1]],
                [- delta * self.c[opponent_idx, 0], delta * self.c[self._player_idx, 0]]
                ]

    self.scores[[self._player_idx, opponent_idx]] += torch.Tensor(r_update)
    self.c[[self._player_idx, opponent_idx]] += torch.Tensor(c_update)

  @property
  def player_score(self) -> float:
    return self.__getitem__(self._player_idx)

  def _apply_rotation(self, opponent_idx: int) -> float:
    z = (self.c[self._player_idx, 0] * self.c[opponent_idx, 1] - self.c[opponent_idx, 0] * self.c[self._player_idx, 1])
    return z

  def __call__(self, opponent_idx: int, beta: float = 1.) -> float:
    z = self._apply_rotation(opponent_idx)
    x = self.player_score - self.__getitem__(opponent_idx) + beta * z
    x = self.alpha * x
    p_hat = torch.nn.Sigmoid()(x)
    assert 0. < p_hat < 1.
    return p_hat
