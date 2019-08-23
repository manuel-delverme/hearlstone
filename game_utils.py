# import baselines.common.running_mean_std
import collections
from typing import Text

import torch

import hs_config
from shared.utils import load_latest_checkpoint


class GameManager(object):
  def __init__(self, seed=None, address=hs_config.Environment.address):
    self.seed = seed
    self._use_heuristic_opponent = True

    self.game_class = hs_config.Environment.get_game_mode(address)
    self.opponents = collections.deque(['random'], maxlen=hs_config.GameManager.max_opponents)
    self._game_score = Elo()

  def update_score(self, score):
    self._game_score.update(score)
    return self._game_score.player_score, self._game_score.games_count

  def opponent_dist(self, n_opponents=None):
    if n_opponents is None:
      n_opponents = len(self.opponents)
    assert n_opponents <= hs_config.GameManager.max_opponents
    return self._game_score.opponent_distribution(n_opponents)

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
      initial_dist = self.opponent_dist(1)
      hs_game.set_opponents(opponents=['default'], opponent_dist=initial_dist)
    else:
      hs_game.set_opponents(opponents=self.opponents, opponent_dist=initial_dist)

    return hs_game

  def add_learned_opponent(self, checkpoint_file: Text):
    assert isinstance(checkpoint_file, str)

    # refresh the queue looking at other files
    if hs_config.Environment.arena:
      for experiment_id in hs_config.Environment.opponent_keys:
        ckpt = load_latest_checkpoint(experiment_id=experiment_id)
        if ckpt is not None and ckpt not in self.opponents:
          assert isinstance(ckpt, str)
          self.opponents.append(ckpt)
          self._game_score.set_from_player(len(self.opponents))

    # replace worst performer with the lastest copy
    worst_opponent_idx = self._game_score.scores[:len(self.opponents)].argmin().item()
    self.opponents.remove(self.opponents[worst_opponent_idx])
    self.opponents.append(checkpoint_file)
    self._game_score.set_from_player(worst_opponent_idx)


class Elo:
  def __init__(self):
    # https://arxiv.org/pdf/1806.02643.pdf

    max_opponents = hs_config.GameManager.max_opponents
    self.max_opponents = max_opponents

    self.games = torch.zeros(max_opponents)
    self.scores = torch.ones((max_opponents + 1,)) * hs_config.GameManager.elo_lr
    self.c = torch.rand(size=(max_opponents + 1, 2))
    self.alpha = hs_config.GameManager.elo_scale
    self.k = hs_config.GameManager.elo_lr
    self._player_idx = -1  # last on the list

  def __getitem__(self, item: int) -> float:
    return self.scores[item]

  def set_from_player(self, idx):
    self.scores[idx] = self.scores[self._player_idx]
    self.c[idx] = self.c[self._player_idx]

  def update(self, scores: dict):
    for idx, score in scores.items():
      self.games[idx] += len(score)
      p = torch.Tensor(score).clamp(0, 1).mean()
      p_hat = self.__call__(idx)
      delta = p - p_hat
      self._grad(delta, opponent_idx=idx)

  def _grad(self, delta: float, opponent_idx: int):
    r_update = (self.k * delta, - self.k * delta)
    c_update = [[delta * self.c[opponent_idx, 1], - delta * self.c[self._player_idx, 1]],
                [- delta * self.c[opponent_idx, 0], delta * self.c[self._player_idx, 0]]
                ]

    self.scores[[self._player_idx, opponent_idx]] = self.scores[[self._player_idx, opponent_idx]] + torch.Tensor(r_update)
    self.c[[self._player_idx, opponent_idx]] = self.c[[self._player_idx, opponent_idx]] + torch.Tensor(c_update)

  @property
  def games_count(self) -> torch.Tensor:
    return self.games

  @property
  def player_score(self) -> float:
    return self.__getitem__(self._player_idx)

  def opponent_distribution(self, opponents=None):
    if opponents is None:
      opponents = self.max_opponents
    # return torch.softmax(self.scores[:opponents], dim=0).numpy()
    p_ij = torch.Tensor([1 - self.__call__(idx) for idx in range(opponents)])
    return (p_ij / p_ij.sum()).numpy()

  def _apply_rotation(self, opponent_idx: int) -> torch.Tensor:
    z = (self.c[self._player_idx, 0] * self.c[opponent_idx, 1] - self.c[opponent_idx, 0] * self.c[self._player_idx, 1])
    return z

  def __call__(self, opponent_idx: int, beta: float = 1.) -> float:
    z = self._apply_rotation(opponent_idx)
    x = self.player_score - self.__getitem__(opponent_idx) + beta * z
    x = self.alpha * x
    p_hat = torch.nn.Sigmoid()(x)
    assert 0. < p_hat < 1.
    return p_hat


def to_prob(r):
  return r / 2 + 0.5
