# import baselines.common.running_mean_std
import collections
import glob
import os
import shutil
from typing import Optional, Text

import numpy as np
import torch

import hs_config


class GameManager(object):
  def __init__(self, seed=None, address=hs_config.Environment.address, max_opponents=hs_config.GameManager.selection_size):

    self.seed = seed
    self._use_heuristic_opponent = True
    self.max_opponents = max_opponents

    self.game_class = hs_config.Environment.get_game_mode(address)

    self.opponents = None
    self.elo = None
    self.ranking = None
    self.model_list = None

  def update_score(self, score):
    self.elo.update(score)
    return self.elo.player_score, self.elo.games_count

  def opponent_dist(self):
    opponent_dist = self.elo.opponent_distribution(number_of_active_opponents=len(self.opponents))
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

      initial_dist = np.ones(shape=(1,))  # sample heuristic with prob 1

      hs_game.set_opponents(opponents=['default'], opponent_dist=initial_dist)
    else:
      hs_game.set_opponents(opponents=self.opponents, opponent_dist=initial_dist)

    return hs_game

  def create_league(self, player_ckpt):
    new_ckpt = os.path.join(os.path.dirname(player_ckpt), hs_config.GameManager.player_fname)

    player_ckpt = shutil.copyfile(player_ckpt, new_ckpt)

    model_list = list(glob.glob(hs_config.GameManager.model_paths))

    if len(model_list) == 0:
      print("[GAME MANAGER] Found 0 opponents. Crushing...")
      raise FileNotFoundError
    else:
      model_list = model_list[:hs_config.GameManager.league_size]
      print(f"[GAME MANAGER] Found {len(model_list)} opponents")

    model_list += [player_ckpt]
    self.model_list = model_list
    self.reset(max_opponents=len(model_list))
    self.opponents.extend(model_list)
    self.use_heuristic_opponent = False

  def set_selection(self):
    self.ranking = collections.OrderedDict({ckpt: score for ckpt, score in zip(self.model_list, self.elo.player_strength().numpy())})
    self.reset(max_opponents=hs_config.GameManager.selection_size)
    self.update_selection()

  def update_selection(self, player_score=None):
    assert self.elo.player_idx in (self.max_opponents, -1)
    if player_score is None:
      self.ranking[hs_config.GameManager.player_fname] = self.elo.player_strength().numpy()[-1]  # this one update the player checkpotin
    else:
      player_score = self.ranking[hs_config.GameManager.player_fname]

    ranking_values = list(self.ranking.values())
    sample_size = min(hs_config.GameManager.selection_size, len(ranking_values))

    p = boltzmann(scores=torch.Tensor(ranking_values), tau=hs_config.GameManager.tau).numpy()
    idxs = np.random.choice(np.arange(0, len(ranking_values)), p=p, replace=False, size=sample_size)
    assert self.model_list == list(self.ranking.keys())
    new_league = [self.model_list[idx] for idx in idxs]
    league_stats = np.array([self.ranking[player] for player in new_league])

    assert self.opponents.maxlen == len(new_league)
    print(f'[GAME MANAGER] New league avg score, {league_stats.mean()}, with std {league_stats.std()}, current_player {player_score}')
    self.opponents.extend(new_league)
    return league_stats

  def add_learned_opponent(self, checkpoint_file: Text):
    assert isinstance(checkpoint_file, str)

    if self.model_list is None:
      self.elo.set_score_from_player(len(self.opponents))
      self.opponents.append(checkpoint_file)
    else:
      new_ckpt = os.path.join(os.path.dirname(checkpoint_file), hs_config.GameManager.player_fname)
      shutil.copyfile(checkpoint_file, new_ckpt)
      self.model_list.append(checkpoint_file)
      self.ranking[checkpoint_file] = self.elo.player_strength().numpy()[-1]
      print(f'[GAME MANAGER] Added learning opponent. Current score {self.elo.player_score}')

  def reset(self, max_opponents=hs_config.GameManager.league_size):
    if self.elo is None:
      self.elo = Elo(max_opponents=max_opponents)
    else:
      self.elo.reset(max_opponents=max_opponents)
    self.opponents = collections.deque(maxlen=max_opponents)
    self.use_heuristic_opponent = True


class Elo:
  def __init__(self, max_opponents: int = hs_config.GameManager.selection_size):
    # https://arxiv.org/pdf/1806.02643.pdf

    self.max_opponents = None
    self.games = None
    self._scores = None
    self._c = None
    self.player_idx = None
    self.alpha = hs_config.GameManager.elo_scale
    self.k = hs_config.GameManager.elo_lr
    self.tau = hs_config.GameManager.tau
    self.reset(max_opponents=max_opponents)

  def __getitem__(self, item: int) -> float:
    return self._scores[item]

  def reset(self, max_opponents: Optional[int] = None):
    self.max_opponents = max_opponents
    self.games = torch.zeros(max_opponents)
    self._scores = torch.ones((max_opponents + 1)) * hs_config.GameManager.elo_lr
    self._c = torch.rand(size=(max_opponents + 1, 2))  # + 1 for the player in self play mode
    self.player_idx = -1

  def set_player_index(self, idx: int):
    assert idx < self.max_opponents + 1
    self.player_idx = idx

  def set_score_from_player(self, idx: int):
    self._scores[idx] = self._scores[self.player_idx]
    self._c[idx] = self._c[self.player_idx]

  def update(self, scores: dict):
    for idx, score in scores.items():
      self.games[idx] += len(score)
      p = torch.Tensor(score).clamp(0, 1).mean()
      p_hat = self.__call__(idx)
      delta = p - p_hat
      self._grad(delta, opponent_idx=idx)

  def _grad(self, delta: float, opponent_idx: int):
    r_update = (self.k * delta, - self.k * delta)
    c_update = [[delta * self._c[opponent_idx, 1], - delta * self._c[self.player_idx, 1]],
                [- delta * self._c[opponent_idx, 0], delta * self._c[self.player_idx, 0]]
                ]

    self._scores[[self.player_idx, opponent_idx]] = self._scores[[self.player_idx, opponent_idx]] + torch.Tensor(r_update)
    self._c[[self.player_idx, opponent_idx]] = self._c[[self.player_idx, opponent_idx]] + torch.Tensor(c_update)

  @property
  def games_count(self) -> torch.Tensor:
    return self.games

  def player_strength(self):
    # the probability of winning against any player in the league
    p = torch.Tensor([self.__call__(idx) for idx in range(self.max_opponents)])
    avg = p.mean()[None]  # the avberage of this is his strength
    return torch.cat([p, avg], dim=0)

  @property
  def player_score(self) -> float:
    return self.__getitem__(self.player_idx)

  def opponent_distribution(self, number_of_active_opponents) -> np.ndarray:
    # TODO try to rewrite this cenetering at 45
    score = self.player_strength()[:number_of_active_opponents]
    prob_losing = 1. - score
    prob_losing[prob_losing > hs_config.GameManager.strength_th] = -1
    prob_losing[prob_losing < (1 - hs_config.GameManager.strength_th)] = -1
    return boltzmann(scores=prob_losing, tau=self.tau).numpy()

  @property
  def scores(self) -> torch.Tensor:
    return self._scores[:-1]

  def _apply_rotation(self, opponent_idx: int) -> torch.Tensor:
    z = (self._c[self.player_idx, 0] * self._c[opponent_idx, 1] - self._c[opponent_idx, 0] * self._c[self.player_idx, 1])
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


def boltzmann(scores, tau=1.):
  assert isinstance(scores, torch.Tensor)  # softmax is location invariant but not scale invariant
  return torch.softmax(tau * scores, dim=0)
