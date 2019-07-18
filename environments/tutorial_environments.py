import collections
import random

import fireplace.cards
import numpy as np

import agents.heuristic.hand_coded
import environments.vanilla_hs
import hs_config


class TradingHS(environments.vanilla_hs.VanillaHS):
  def __init__(self, level=hs_config.VanillaHS.level, seed=None, extra_seed=None):

    if level == 0:
      self.player_board = [fireplace.cards.filter(name="Bloodfen Raptor", collectible=True), ] * 1
      self.opponent_board = [fireplace.cards.filter(name="Wisp", collectible=True), ] * 1  # Wisp
    elif level == 1:
      self.player_board = [fireplace.cards.filter(name="Bloodfen Raptor", collectible=True), ] * 7
      self.opponent_board = [fireplace.cards.filter(name="Wisp", collectible=True), ] * 7  # Wisp
    elif level == 3:
      self.player_board = [
        fireplace.cards.filter(name="Bloodfen Raptor", collectible=True),
        fireplace.cards.filter(name="Bloodfen Raptor", collectible=True),
        fireplace.cards.filter(name="Wisp", collectible=True),
        fireplace.cards.filter(name="Wisp", collectible=True),
        fireplace.cards.filter(name="Murloc Raider", collectible=True),
        fireplace.cards.filter(name="Bloodfen Raptor", collectible=True),
      ]
      self.opponent_board = [
        fireplace.cards.filter(name="Chillwind Yeti", collectible=True),
        fireplace.cards.filter(name="Am'gam Rager", collectible=True),
        fireplace.cards.filter(name="Magma Rager", collectible=True),
        fireplace.cards.filter(name="Magma Rager", collectible=True),
      ]
    elif level == 4:
      self.generate_board()
    else:
      raise ValueError
    super(TradingHS, self).__init__(
      max_cards_in_hand=0,
      skip_mulligan=True,
      starting_hp=hs_config.VanillaHS.starting_hp,
      sort_decks=hs_config.VanillaHS.sort_decks,
    )
    self.level = level

    immunity = fireplace.cards.utils.buff(immune=True)
    self.simulation.player.hero.set_current_health(hs_config.VanillaHS.starting_hp)
    self.simulation.player.hero.tags.update(immunity.tags)
    self.simulation.opponent.hero.set_current_health(hs_config.VanillaHS.starting_hp)
    self.simulation.opponent.hero.tags.update(immunity.tags)
    self.opponent = agents.heuristic.hand_coded.PassingAgent()
    self.minions_in_board = level

  def generate_game(self):
    pass

  @classmethod
  def generate_board(cls):
    def encode(card):
      return card.atk, card.health

    def exit_condition(opponent_hp_left, player_hp_left):
      return opponent_hp_left > 0 and player_hp_left > 0

    actions = []

    player_board = tuple()
    opponent_board = tuple()
    player_board = tuple(encode(fireplace.cards.db[c]) for c, in player_board)
    opponent_board = tuple(encode(fireplace.cards.db[c]) for c, in opponent_board)

    for atk_idx in range(10):
      for def_idx in range(10):
        actions.append(make_attack(atk_idx, def_idx, sign=+1))
    actions = tuple(actions)
    return cls.bf_search(actions, exit_condition, opponent_board, player_board)

  @classmethod
  def solve_optimally(cls, player_board, opponent_board):
    def encode(card):
      return card.atk, card.health

    def exit_condition(opponent_hp_left):
      return opponent_hp_left <= 0

    actions = []

    player_board = tuple(encode(fireplace.cards.db[c]) for c, in player_board)
    opponent_board = tuple(encode(fireplace.cards.db[c]) for c, in opponent_board)

    for atk_idx, src in enumerate(player_board):
      for def_idx, target in enumerate(opponent_board):
        actions.append(make_attack(atk_idx, def_idx, sign=-1))
    actions = tuple(actions)
    return cls.bf_search(actions, exit_condition, opponent_board, player_board)

  @classmethod
  def bf_search(cls, actions, exit_condition, opponent_board, player_board):
    PLAYER, OPPONENT = 0, 1
    ATK, HP = 0, 1
    open_nodes = [(player_board, opponent_board), ]
    forbidden_edges = collections.defaultdict(set)
    closed = set()
    best_solution = -1
    while open_nodes:
      node1 = open_nodes.pop(0)
      if node1 in closed:
        continue

      for edge_idx, edge in enumerate(actions):
        if edge_idx in forbidden_edges[node1]:
          continue

        try:
          node2 = edge(node1)
        except IndexError:
          continue
        else:
          if node2 not in closed:
            forbidden_edges[node2].update(forbidden_edges[node1])
            forbidden_edges[node2].add(edge_idx)

            player_hp_left = sum(max(c[HP], 0) for c in node2[PLAYER])
            opponent_hp_left = sum(max(c[HP], 0) for c in node2[OPPONENT])

            if player_hp_left > 0:
              if exit_condition(opponent_hp_left, player_hp_left):
                best_solution = max(player_hp_left, best_solution)

              if opponent_hp_left:
                open_nodes.append(node2)

      closed.add(node1)
    return best_solution

  def reinit_game(self):
    super(TradingHS, self).reinit_game()
    self.generate_board()

    for minion in self.player_board:
      self.simulation.player.summon(minion)

    for minion in self.opponent_board:
      self.simulation.opponent.summon(minion)

  def gather_transition(self, autoreset):
    game_observation, reward, terminal, info = super(TradingHS, self).gather_transition(autoreset)
    num_player_minions = len(self.simulation.player.characters[1:])
    num_opponent_minions = len(self.simulation.opponent.characters[1:])
    if num_player_minions == 0 or num_opponent_minions == 0 or self.simulation.game.turn == hs_config.VanillaHS.max_turns:
      terminal = True

    if terminal:
      info['game_statistics'] = {
        'num_games': self.games_played,
        'num_steps': self.episode_steps,
        'turn': self.simulation.game.turn,
        'outcome': reward,
      }
      if autoreset:
        new_obs, _, _, new_info = self.reset()
        game_observation = new_obs
        info['possible_actions'] = new_info['possible_actions']
        info['observation'] = new_info['observation']

    # if autoreset and terminal:
    #   new_obs, _, _, new_info = self.reset()
    #   game_observation = new_obs
    #   info['possible_actions'] = new_info['possible_actions']
    #   info['observation'] = new_info['observation']

    return game_observation, reward, terminal, info

  def calculate_reward(self):
    num_opponent_minions = len(self.simulation.opponent.characters[1:])
    num_player_minions = len(self.simulation.player.characters[1:])
    # if num_player_minions > 0 and num_opponent_minions > 0:
    #   return -1

    if num_opponent_minions > 0:
      reward = 0
    else:
      reward = num_player_minions
    return np.array(reward, dtype=np.float32)

  def __str__(self):
    return 'TradingHS:{}'.format(self.level, )


def make_attack(atk_idx, def_idx, sign=-1):
  def attack(node):
    player_board, opponent_board = node

    if atk_idx == len(player_board):
      player_board = (*player_board, (random.randint(0, 10), 0))

    if def_idx == len(opponent_board):
      opponent_board = (*opponent_board, (random.randint(0, 10), 0))

    attacker = list(player_board[atk_idx])
    defender = list(opponent_board[def_idx])

    ATK, HP = 0, 1

    attacker[HP] += sign * defender[ATK]
    defender[HP] += sign * attacker[ATK]

    player_board = (*player_board[:atk_idx], tuple(attacker), *player_board[atk_idx + 1:])
    opponent_board = (*opponent_board[:def_idx], tuple(defender), *opponent_board[def_idx + 1:])

    return player_board, opponent_board

  return attack


t = TradingHS()
t.reinit_game()
