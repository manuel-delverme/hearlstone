import collections
from typing import Text

import numpy as np
import torch

import environments.sabber2_hs
import environments.sabber_hs
import environments.trading_hs
import hs_config
import shared.constants as C


class GameManager(object):
  def __init__(self, seed=None):
    self.seed = seed
    self._use_heuristic_opponent = True

    if hs_config.Environment.game_mode == C.GameModes.TradingHS:
      self.game_class = environments.trading_hs.TradingHS
    elif hs_config.Environment.game_mode == C.GameModes.SabberHS_rpc:
      self.game_class = environments.sabber_hs.Sabberstone
    elif hs_config.Environment.game_mode == C.GameModes.SabberHS_mmf:
      self.game_class = environments.sabber2_hs.Sabberstone2
    else:
      raise ValueError

    self.opponents = collections.deque(['random'], maxlen=hs_config.GameManager.max_opponents)
    self.ladder = Ladder()

    self.observation_space = self.game_class.observation_space
    self.action_space = self.game_class.action_space

  def update_score(self, score):
    self.ladder.update(score)
    return self.ladder.player_score, self.ladder.games_count

  def opponent_dist(self):
    opponent_dist = self.ladder.opponent_distribution(number_of_active_opponents=len(self.opponents))
    return opponent_dist

  @property
  def use_heuristic_opponent(self):
    return self._use_heuristic_opponent

  @use_heuristic_opponent.setter
  def use_heuristic_opponent(self, value):
    self._use_heuristic_opponent = value

  def __call__(self, env_number):
    hs_game = self.game_class(env_number=env_number)
    initial_dist = self.opponent_dist()
    if self.use_heuristic_opponent:

      initial_dist = torch.ones(size=(1,)).numpy()

      hs_game.set_opponents(opponents=['default'], opponent_dist=initial_dist)
    else:
      hs_game.set_opponents(opponents=self.opponents, opponent_dist=initial_dist)

    return hs_game

  def add_learned_opponent(self, checkpoint_file: Text):
    assert isinstance(checkpoint_file, str)
    last_opponent = len(self.opponents)
    self.ladder.set_score_from_player(last_opponent)

    self.opponents.append(checkpoint_file)


class Ladder:
  def __init__(self):
    # https://arxiv.org/pdf/1806.02643.pdf

    max_opponents = hs_config.GameManager.max_opponents

    self.games = torch.zeros(max_opponents)
    self._scores = torch.ones((max_opponents + 1,)) * hs_config.GameManager.elo_lr
    self._c = torch.rand(size=(max_opponents + 1, 2))
    self.alpha = hs_config.GameManager.elo_scale
    self.k = hs_config.GameManager.elo_lr
    self.player_idx = -1  # last on the list
    self.tau = hs_config.GameManager.tau
    self.max_opponents = max_opponents

  def __getitem__(self, item: int) -> torch.Tensor:
    return self.scores[item]

  def update(self, scores: dict):
    for idx, score in scores.items():
      p = torch.Tensor(score).clamp(0, 1).mean()
      p_hat = self.__call__(idx)
      delta = p - p_hat
      self._grad(delta, opponent_idx=idx)

  def _grad(self, delta: float, opponent_idx: int):
    r_update = (self.k * delta, - self.k * delta)
    c_update = [[delta * self._c[opponent_idx, 1], - delta * self._c[self.player_idx, 1]],
                [- delta * self._c[opponent_idx, 0], delta * self._c[self.player_idx, 0]]
                ]

    self.scores[[self.player_idx, opponent_idx]] = self.scores[[self.player_idx, opponent_idx]] + torch.Tensor(r_update)
    self._c[[self.player_idx, opponent_idx]] = self._c[[self.player_idx, opponent_idx]] + torch.Tensor(c_update)

  def set_score_from_player(self, idx):
    self._scores[idx] = self._scores[self.player_idx]
    self._c[idx] = self._c[self.player_idx]

  @property
  def games_count(self) -> torch.Tensor:
    return self.games

  def player_strength(self):
    # the probability of winning against any player in the league
    p = torch.Tensor([self.__call__(idx) for idx in range(self.max_opponents)])
    avg = p.mean()[None]  # the avberage of this is his strength
    return torch.cat([p, avg], dim=0)

  @property
  def player_score(self) -> torch.Tensor:
    return self.__getitem__(self.player_idx)

  def opponent_distribution(self, number_of_active_opponents) -> np.ndarray:
    return boltzmann(scores=self.scores[:number_of_active_opponents], tau=self.tau).numpy()

  @property
  def scores(self) -> torch.Tensor:
    return self._scores[:-1]

  def _apply_rotation(self, opponent_idx: int) -> torch.Tensor:
    z = (self._c[self.player_idx, 0] * self._c[opponent_idx, 1] - self._c[opponent_idx, 0] * self._c[self.player_idx, 1])
    return z

  def __call__(self, opponent_idx: int, cyclic_weight: float = hs_config.GameManager.cyclic_weight) -> float:
    z = self._apply_rotation(opponent_idx)
    x = self.player_score - self.__getitem__(opponent_idx) + cyclic_weight * z
    x = self.alpha * x
    p_hat = torch.nn.Sigmoid()(x)
    assert 0. < p_hat < 1.
    return p_hat


def to_prob(r):
  return r / 2 + 0.5


def boltzmann(scores, tau=1.):
  assert isinstance(scores, torch.Tensor)
  return torch.softmax(tau * scores, dim=0)
