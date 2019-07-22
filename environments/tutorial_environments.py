import collections
import random

import fireplace.cards
import fireplace.cards.utils
import fireplace.dsl
import numpy as np

import environments.vanilla_hs
import hs_config


class TradingHS(environments.vanilla_hs.VanillaHS):
  def __init__(self, level=hs_config.Environment.level, seed=None, extra_seed=None):
    super(TradingHS, self).__init__(
      max_cards_in_hand=0,
      skip_mulligan=True,
      starting_hp=hs_config.Environment.starting_hp,
      sort_decks=hs_config.Environment.sort_decks,
    )
    self.level = level
    self.opponent = hs_config.Environment.get_opponent()
    self.minions_in_board = level

  def generate_board(self, acceptable_minions):
    def exit_condition(opponent_hp_left, player_hp_left, node):
      return opponent_hp_left > 0 and player_hp_left > 0 and len(node[0]) + len(node[1]) in acceptable_minions

    def prune_condition(opponent_hp_left, player_hp_left, node):
      return len(node[0]) + len(node[1]) > max(acceptable_minions)

    actions = [make_attack(atk_idx, def_idx, sign=+1) for atk_idx in range(10) for def_idx in range(10)]
    player_board, opponent_board = tuple(), tuple()

    # generate 1 turn strategy
    # print('step 1')
    (player_board, opponent_board), _ = bf_search(actions, exit_condition, prune_condition, player_board,
                                                  opponent_board)
    # print('step 2')
    (player_board, opponent_board), _ = bf_search(actions, exit_condition, prune_condition, player_board,
                                                  opponent_board)
    # print('step -1')
    player_board = (*player_board, (random.randint(1, 3), random.randint(1, 3)))
    self.player_board, self.opponent_board = player_board, opponent_board

  def reinit_game(self):
    super(TradingHS, self).reinit_game()

    if self.level == 0:
      self.player_board = [fireplace.cards.filter(name="Bloodfen Raptor", collectible=True), ] * 1
      self.opponent_board = [fireplace.cards.filter(name="Wisp", collectible=True), ] * 1  # Wisp
    elif self.level == 1:
      self.player_board = [fireplace.cards.filter(name="Bloodfen Raptor", collectible=True), ] * 7
      self.opponent_board = [fireplace.cards.filter(name="Wisp", collectible=True), ] * 7  # Wisp
    elif self.level == 3:
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
    elif self.level == 4:
      self.generate_board(acceptable_minions=(4, 5))  # 2 could be created at the same time
      # assert len(self.opponent_board) <= len(self.player_board)
    else:
      raise ValueError

    stat_stick, = fireplace.cards.filter(name="The Ancient One", collectible=False)

    for minion in self.player_board:
      if isinstance(minion, str):
        self.simulation.player.summon(minion)
      else:
        self.summon_minion(self.simulation.player, minion, stat_stick)

    for minion in self.opponent_board:
      if isinstance(minion, str):
        self.simulation.opponent.summon(minion)
      else:
        self.summon_minion(self.simulation.opponent, minion, stat_stick)

  def summon_minion(self, player, minion, stat_stick):
    player.summon(stat_stick)
    minion_to_handle = player.characters[-1]
    # print(minion_to_handle.exhausted)
    MOONFIRE = "CS2_008"

    actor = player.game.current_player
    old_cards_in_hand = actor.max_hand_size
    actor.max_hand_size = 1

    while minion_to_handle.health > minion[1]:
      card = actor.give(MOONFIRE)
      card.play(target=minion_to_handle)

    actor.max_hand_size = old_cards_in_hand
    minion_to_handle.atk = minion_to_handle._atk = minion[0]
    minion_to_handle.charge = True

  def gather_transition(self, autoreset):
    game_observation, reward, terminal, info = super(TradingHS, self).gather_transition(autoreset)
    num_player_minions = len(self.simulation.player.characters[1:])
    num_opponent_minions = len(self.simulation.opponent.characters[1:])

    # info['possible_actions'][0] = 0

    if num_player_minions == 0:
      terminal = True
    if num_opponent_minions == 0:
      terminal = True
    if self.simulation.game.turn == hs_config.Environment.max_turns:
      terminal = True
    # if info['possible_actions'].max(0) == 0:
    #   terminal = True

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
      player_board = (*player_board, (random.randint(1, 10), 0))

    if def_idx == len(opponent_board):
      opponent_board = (*opponent_board, (random.randint(1, 10), 0))

    attacker = list(player_board[atk_idx])
    defender = list(opponent_board[def_idx])

    ATK, HP = 0, 1

    attacker[HP] += sign * defender[ATK]
    if 0 > attacker[HP] or attacker[HP] > 10:
      raise ValueError

    defender[HP] += sign * attacker[ATK]
    if 0 > defender[HP] or defender[HP] > 10:
      raise ValueError

    player_board = (*player_board[:atk_idx], tuple(attacker), *player_board[atk_idx + 1:])
    opponent_board = (*opponent_board[:def_idx], tuple(defender), *opponent_board[def_idx + 1:])

    return player_board, opponent_board

  return attack


def bf_search(actions, exit_condition, prune_condition, player_board, opponent_board):
  PLAYER, OPPONENT = 0, 1
  ATK, HP = 0, 1
  open_nodes = [(player_board, opponent_board), ]
  forbidden_edges = collections.defaultdict(set)
  sequences = collections.defaultdict(set)

  closed = set()
  best_solution = -1

  while open_nodes:
    node1 = open_nodes.pop(0)
    if node1 in closed:
      continue

    random.shuffle(actions)
    for edge_idx, edge in enumerate(actions):
      if edge_idx in forbidden_edges[node1]:
        continue

      try:
        node2 = edge(node1)
      except IndexError:
        # if hasattr(edge, 'idx'):
        #   print('IndexError', node1, edge.idx)
        continue
      except ValueError:
        # if hasattr(edge, 'idx'):
        #   print('ValueError', node1, edge.idx)
        continue
      else:
        if node2 in closed:
          continue

        # print(node1[0], 'vs', node1[1], '->', node2[0], 'vs', node2[1])
        forbidden_edges[node2].update(forbidden_edges[node1])
        forbidden_edges[node2].add(edge_idx)

        player_hp_left = sum(max(c[HP], 0) for c in node2[PLAYER])
        opponent_hp_left = sum(max(c[HP], 0) for c in node2[OPPONENT])

        if player_hp_left > 0:
          if exit_condition(opponent_hp_left, player_hp_left, node2):
            break

          if opponent_hp_left and not prune_condition(opponent_hp_left, player_hp_left, node2):
            open_nodes.append(node2)

    closed.add(node1)
  return node2, forbidden_edges[node2]
