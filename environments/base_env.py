import collections
import time
from abc import ABC, abstractmethod
from enum import IntEnum

import gym
import numpy as np
import torch

import hs_config
import shared.constants as C


class BaseEnv(gym.Env, ABC):
  class GameActions(IntEnum):
    PASS_TURN = 0

  def __init__(self):
    self.opponent = None
    self.deterministic_opponent = None
    self.opponent_dist = None
    self.opponents = [None, ]
    self.opponents_lookup = {}

  @abstractmethod
  def agent_game_vale(self):
    raise NotImplemented

  def set_opponents(self, opponents, opponent_dist, deterministic_opponent=True):
    _opponents = []
    for player_hash in opponents:
      if player_hash not in self.opponents_lookup:
        self.opponents_lookup[player_hash] = self.load_opponent(player_hash)

      _opponents.append(self.opponents_lookup[player_hash])

    self.opponents = _opponents
    self.opponent_dist = opponent_dist
    self.deterministic_opponent = deterministic_opponent

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
    checkpoint = torch.load(checkpoint_file)
    opponent_network = checkpoint['network']
    assert isinstance(opponent_network, agents.learning.models.randomized_policy.ActorCritic), opponent_network
    opponent = agents.learning.ppo_agent.PPOAgent(opponent_network.num_inputs, opponent_network.num_possible_actions, device='cpu',
                                                  experiment_id=None)

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
  hand_encoding_size = len(C.Card._fields)  # atk, health, exhaust
  hero_encoding_size = len(C.Hero._fields)  # atk, health, exhaust, hero_power
  minion_encoding_size = len(C.Minion._fields)  # atk, health, exhaust

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
      offset, board, hand, mana, hero, board_size = self.render_player(obs)
      offset += 30 * C.INACTIVE_CARD_ENCODING_SIZE  # SKIP SHOWING THE DECK

      self.gui.draw_agent(hero=hero, board=board, hand=hand, mana=mana)
      hero_health = hero.health

      offset, board, hand, mana, hero, board_size = self.render_player(obs, offset, show_hand=False)
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

      row_number += 1
      self.gui.log(f"value: {float(value)}, choice: {int(choice)}", row=row_number, multiline=False)
      row_number += 1

      first_row_number = row_number
      max_rows = self.gui.game_height - first_row_number - 6 - 16
      if len(pretty_actions) > max_rows:
        pretty_actions = pretty_actions[:max_rows]

      # self.gui.log(f"{[p[1] for p in pretty_actions[:3]]}", row=row_number, multiline=True)
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
      time.sleep(3)
      self.gui.screen.nodelay(True)
      self.health.clear()
      self.values.clear()
      for _ in range(1000):
        if -1 == self.gui.screen.getch():
          break
      else:
        raise TimeoutError('failed to flush the char buffer')
      self.gui.screen.nodelay(False)
      self.gui.log("Press key to continue", row=2)

    if mode == 'human':
      return self.gui.screen.getch()
    else:
      raise NotImplementedError

  def render_player(self, obs, offset=0, show_hand=True):
    mana = obs[offset]
    offset += 1

    hero = obs[offset: offset + self.hero_encoding_size]
    hero = C.Hero(*hero)
    offset += self.hero_encoding_size

    hand = []
    if show_hand:
      for minion_number in range(hs_config.Environment.max_cards_in_hand):
        card_obs = obs[offset: offset + self.hand_encoding_size]
        if card_obs.max() > -1:
          card = C.Card(*card_obs)
          if card.atk == -1 and card.health == -1:  # it's a spell
            card_dict = card._asdict()
            spell = C.SPELLS(C.REVERSE_CARD_LOOKUP[tuple(card_obs)])

            card_id = ''.join(i for i in str(spell)[7:] if i.isupper())
            if len(card_id) == 1:
              card_id = str(spell)[7:9]
            card_dict['atk'] = card_id
            card_dict['health'] = card.cost

            card = C.Card(**card_dict)

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

    board_size = obs[offset]
    offset += 1
    return offset, board, hand, mana, hero, board_size

  def parse_options(self, game_snapshot, return_options):
    raise NotImplementedError
