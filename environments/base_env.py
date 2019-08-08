import collections
import math
import time
from abc import ABC, abstractmethod
from enum import IntEnum

import gym
import numpy as np
import torch

import hs_config
import shared.constants as C


# source: https://github.com/kroitor/asciichart/blob/master/asciichartpy/__init__.py


def plot(series):
  minimum = min(series)
  maximum = max(series)

  interval = abs(float(maximum) - float(minimum))
  interval = interval or 1.0  # avoid division by 0
  offset = 3
  height = interval
  ratio = height / interval
  min2 = math.floor(float(minimum) * ratio)
  max2 = math.ceil(float(maximum) * ratio)

  intmin2 = int(min2)
  intmax2 = int(max2)

  rows = abs(intmax2 - intmin2) or 1
  width = len(series) + offset

  result = [[' '] * width for i in range(rows + 1)]

  # Axis and labels.
  for y in range(intmin2, intmax2 + 1):
    label = '{:8.2f}'.format(float(maximum) - ((y - intmin2) * interval / rows))
    result[y - intmin2][max(offset - len(label), 0)] = label
    result[y - intmin2][offset - 1] = '┼' if y == 0 else '┤'

  y0 = int(series[0] * ratio - min2)
  result[rows - y0][offset - 1] = '┼'  # first value

  for x in range(0, len(series) - 1):  # plot the line
    y0 = int(round(series[x + 0] * ratio) - intmin2)
    y1 = int(round(series[x + 1] * ratio) - intmin2)
    if y0 == y1:
      result[rows - y0][x + offset] = '─'
    else:
      result[rows - y1][x + offset] = '╰' if y0 > y1 else '╭'
      result[rows - y0][x + offset] = '╮' if y0 > y1 else '╯'
      start = min(y0, y1) + 1
      end = max(y0, y1)
      for y in range(start, end):
        result[rows - y][x + offset] = '│'

  return '\n'.join([''.join(row) for row in result])


class BaseEnv(gym.Env, ABC):
  class GameActions(IntEnum):
    PASS_TURN = 0

  class GameOver(Exception):
    pass

  def __init__(self):
    self.opponent = None
    self.opponents = [None, ]
    self.opponents_lookup = {}

  @abstractmethod
  def agent_game_vale(self):
    raise NotImplemented

  def set_opponents(self, opponents):
    _opponents = []
    for player_hash in opponents:
      if player_hash not in self.opponents_lookup:
        self.opponents_lookup[player_hash] = self.load_opponent(player_hash)

      _opponents.append(self.opponents_lookup[player_hash])

    self.opponents = _opponents

  def load_opponent(self, checkpoint_file):
    if checkpoint_file == 'default':
      return hs_config.Environment.get_opponent()()
    if checkpoint_file == 'random':
      import agents.heuristic.random_agent
      return agents.heuristic.random_agent.RandomAgent()

    import agents.learning.ppo_agent

    # if self.use_heuristic_opponent:
    #   assert isinstance(self.opponents[0], agents.heuristic.random_agent.RandomAgent)
    #   self.opponents = []
    #   self.use_heuristic_opponent = False
    opponent_network, = torch.load(checkpoint_file)
    assert isinstance(opponent_network, agents.learning.models.randomized_policy.ActorCritic), opponent_network
    opponent = agents.learning.ppo_agent.PPOAgent(opponent_network.num_inputs, opponent_network.num_possible_actions, device='cpu')

    del opponent.pi_optimizer
    del opponent.value_optimizer
    opponent_network.eval()
    for network in (opponent_network.actor, opponent_network.critic):
      for param in network.parameters():
        param.requires_gradient = False

    opponent.actor_critic = opponent_network
    if hasattr(opponent, 'logger'):  # the logger has rlocks which cannot change process so RIP
      del opponent.logger  # TODO: this is an HACK, refactor it away
    if hasattr(opponent, 'timer'):  # the timer has rlocks which cannot change process so RIP
      del opponent.timer  # TODO: this is an HACK, refactor it away

    return opponent


class RenderableEnv(BaseEnv):
  hand_encoding_size = None  # atk, health, exhaust
  hero_encoding_size = None  # atk, health, exhaust, hero_power
  minion_encoding_size = None  # atk, health, exhaust

  def __init__(self):
    super().__init__()
    self.gui = None
    self.last_info = None
    self.last_observation = None
    self.values = collections.deque(maxlen=100)
    self.health = collections.deque(maxlen=100)

  def render(self, mode='human', choice=None, action_distribution=None, value=None, reward=None):
    if self.gui is None:
      import gui
      self.gui = gui.GUI()

    if value is not None:
      self.values.append(float(value))

    if reward is None:
      obs = self.last_observation
      offset, board, hand, mana, hero = self.render_player(obs)
      self.gui.draw_agent(hero=hero, board=board, hand=hand, mana=mana)
      hero_health = hero.health

      offset, board, hand, mana, hero = self.render_player(obs, offset, show_hand=False)
      self.gui.draw_opponent(hero=hero, board=board, hand=hand, mana=mana)
      self.health.append(float(hero_health) / float(hero.health))

      info = self.last_info.copy()
      pi = np.argwhere(info['possible_actions']).flatten()
      pretty_actions = []
      options, _ = self.parse_options(self.game_snapshot, return_options=True)

      logit = {}
      for possible_idx in pi:
        logit[possible_idx] = action_distribution.log_prob(torch.tensor(possible_idx))

      lk = logit.keys()
      lv = [logit[k] for k in lk]

      if len(lv) == 1:
        probs = [1]
      else:
        _probs = torch.tensor(lv).softmax(dim=0)
        probs = dict(zip(lk, _probs))

      for possible_idx in pi:
        action_log_prob = action_distribution.log_prob(torch.tensor(possible_idx))
        pretty_actions.append(
            (possible_idx, options[possible_idx].print, float(action_log_prob), float(probs[possible_idx])))

      row_number = 1
      self.gui.log(str(self.__str__()), row=row_number, multiline=False)

      values = np.array(self.values)
      mu, std = values.mean(), values.std()
      value_norm = np.linalg.norm(values)
      values = values / value_norm

      row_number += 1
      self.gui.log(
          f"value: {float(value):.4f}, choice: {int(choice)}, ", row=row_number, multiline=False)
      row_number += 1
      self.gui.log(f"value norm:{float(value_norm):.4f}, mu:{float(mu):.4f}, std:{float(std):.4f}", row=row_number, multiline=False)
      row_number += 1
      row_number = self.log_plot(row_number, values)
      #row_number = self.log_plot(row_number, self.health)

      first_row_number = row_number
      if len(pretty_actions) > self.gui.game_height - first_row_number - 6 - 16:
        self.gui.log(f"{[p[1] for p in pretty_actions[:3]]}", row=row_number, multiline=True)
      else:
        try:
          for row_number, (k, v, logit, p) in enumerate(sorted(pretty_actions, key=lambda r: r[-2], reverse=True),
                                                        start=row_number):
            if k == choice:
              self.gui.log(f"{k}\t{logit:.1f}\t{p}\t{v} SELECTED", row=row_number, multiline=True)
            else:
              self.gui.log(f"{k}\t{logit:.1f}\t{p}\t{v}", row=row_number, multiline=True)
        except Exception as e:
          print(f" {self.gui.game_height - len(pretty_actions) - first_row_number} " * 1000)
          with open('/tmp/err.log', 'w') as f:
            f.write(str(e))
          time.sleep(1)
    else:
      winner_player = C.AGENT_ID if reward > 0 else C.OPPONENT_ID
      self.gui.windows[C.Players.LOG].clear()
      self.gui.log(f"Game Over. P {winner_player}, reward {reward.item()}", row=1)
      time.sleep(4)
      self.gui.screen.nodelay(True)
      for _ in range(1000):
        if -1 == self.gui.screen.getch():
          break
      else:
        raise TimeoutError('failed to flush the char buffer')
      self.gui.screen.nodelay(False)

      self.gui.log("Waiting for next game", row=2)

    if mode == 'human':
      return self.gui.screen.getch()
    else:
      raise NotImplementedError

  def log_plot(self, row_number, values):
    figure = plot(list(values)[::-1])
    self.gui.log(figure[1:], row=row_number, multiline=True)
    row_number += figure.count('\n') + 2
    return row_number

  def render_player(self, obs, offset=0, show_hand=True):
    mana = obs[offset]
    offset += 1

    hero = obs[offset: offset + self.hero_encoding_size]
    hero = C.Hero(*hero)
    offset += self.hero_encoding_size

    hand = []
    if show_hand:
      for minion_number in range(hs_config.Environment.max_cards_in_hand):
        card = obs[offset: offset + self.hand_encoding_size]
        if card.max() > -1:
          assert card.min() > -1
          card = C.Card(*card)
          hand.append(card)
        offset += self.hand_encoding_size
    # DO NOT TURN INTO ONE STEP NUMPY, flexible > slow
    board = []
    for minion_number in range(hs_config.Environment.max_cards_in_board):
      card = obs[offset: offset + self.minion_encoding_size]
      if card.max() > -1:
        minion = C.Minion(*card)
        board.append(minion)
      offset += self.minion_encoding_size
    return offset, board, hand, mana, hero
