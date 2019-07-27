import collections
import random
from typing import Dict, Text, Any

import numpy as np
import torch
from hearthstone.enums import CardType

import agents.base_agent
import agents.heuristic.random_agent
import hs_config
import specs

# the coin 1746
# Arcane Missels 3 damage randomly 564
# Mirror Image summon 2 minons with taunt 1084
# Arcane Explosion 1 damage all minion  447
# Frostbolt 3 damage + freeze 662
# Arcane Intellect draw 2 cards 555
# Frost Nova freeze all minionk 587
# Fireball 6 damage 315
# Polymoroph transform a minion in a ship 77
# FlameStrike 4 damage to all minion 1004

from collections import OrderedDict

SPELLS = OrderedDict({
  77: 'Polymorph', # transform in a sheep
  315: 'Fireball', # 6 damage
  447: 'Arcane Explosion', # 1 damage all
  555: 'Arcane Intellect', # 2 cards
  587: 'Frost nova', # freeze all
  564: 'Arcane Missels', #  3 random damage
  662: 'Frostbolt', # 3 damage  + freze
  1004: 'Flamestrike', # 4 damage all minions
  1084: 'Mirror Image', # summon two minions
  1746: 'The Coin', #  + 1 mana
})

from environments.sabber_hs import PlayerTaskType, parse_hero, parse_card, parse_minion


class PassingAgent(agents.base_agent.Agent):
  def __init__(self):
    super().__init__()

  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    return 0


class HeuristicAgent(agents.base_agent.Bot):
  def __init__(self, level: int = hs_config.Environment.level):
    assert -1 < level < 7
    super().__init__()
    self.randomness = [
      None,  # 0 is passing
      1,  # 1 is always random
      0.75,  # 2
      0.5,  # 3
      0.25,  # 4
      0.125,  # 5
      0,  # 6 is no random
    ][level]
    self.level = level
    self.random_agent = agents.heuristic.random_agent.RandomAgent()
    self.passing_agent = PassingAgent()

  def _choose(self, observation: np.ndarray, encoded_info: specs.Info):
    if self.level == 0:
      return self.passing_agent.choose(observation, encoded_info)

    if random.random() < self.randomness:
      return self.random_agent.choose(observation, encoded_info)

    possible_actions = encoded_info['original_info']['possible_actions']
    assert encoded_info['possible_actions'].size(0) == 1

    if len(possible_actions) == 1:
      selected_action = possible_actions[0]
    else:
      actions = []
      values = collections.defaultdict(int)
      end_turn = None
      for action in possible_actions:
        if action.card is not None:
          actions.append(action)
          if 'target' not in action.params or action.params['target'] is None:
            # play card
            values[action] += action.card.cost
          else:
            target = action.params['target']
            attacker = action.card

            atk_dies = attacker.health <= target.atk
            def_dies = target.health <= attacker.atk
            if 'HERO' in target.id:
              atk_dies = False
              is_hero = True
            else:
              is_hero = False

            if def_dies and is_hero:
              values[action] += float('inf')
            elif atk_dies and def_dies:
              values[action] += target.cost - attacker.cost
            elif atk_dies and not def_dies:
              values[action] -= float('inf')
            elif not atk_dies and def_dies:
              overkill = target.health / attacker.atk
              if overkill > 2 and not is_hero:
                values[action] = target.cost
            elif not atk_dies and not def_dies:
              values[action] += attacker.atk / target.health
        else:
          end_turn = action

      actions = sorted(actions, key=lambda x: values[x], reverse=True)
      if values[actions[0]] < 0:
        selected_action = end_turn
      else:
        selected_action = actions[0]

    for enc_action, action_obj in zip(torch.nonzero(encoded_info['possible_actions']),
                                      encoded_info['original_info']['possible_actions']):
      if action_obj == selected_action:
        return enc_action[1]
    else:
      raise Exception


def parse_game(info):
  observation = info['observation']

  player_hero = Hero(*parse_hero(observation.CurrentPlayer.hero))
  opponent_hero = Hero(*parse_hero(observation.CurrentOpponent.hero))

  hand_zone = list(map(lambda x: Card(*parse_card(x)), observation.CurrentPlayer.hand_zone.entities))
  player_board = list(map(lambda x: Minion(*parse_minion(x)), observation.CurrentPlayer.board_zone.minions))
  opponent_board = list(map(lambda x: Minion(*parse_minion(x)), observation.CurrentOpponent.board_zone.minions))
  return player_hero, opponent_hero, hand_zone, player_board, opponent_board


Minion = collections.namedtuple('minion', ['id', 'atk', 'health'])
Card = collections.namedtuple('card', ['id', 'atk', 'base_health', 'cost'])
Hero = collections.namedtuple('hero', ['id', 'atk', 'health', 'exhausted'])


class SabberAgent(HeuristicAgent):
  def __init__(self, level: int = hs_config.Environment.level):
    super(SabberAgent, self).__init__(level)

  def _choose(self, observation: np.ndarray, encoded_info: specs.Info):
    if self.level == 0:
      return self.passing_agent.choose(observation, encoded_info)

    if random.random() < self.randomness:
      return self.random_agent.choose(observation, encoded_info)

    possible_actions = encoded_info['original_info']['possible_actions']
    player_hero, opponent_hero, hand_zone, player_board, opponent_board = parse_game(encoded_info['original_info'])

    cost = lambda x: (x.atk + x.health) / 2

    if len(possible_actions) == 1:
      selected_action = 0
    else:
      actions = []
      values = collections.defaultdict(int)
      end_turn = None
      for idx, action in possible_actions.items():
        if action.type != PlayerTaskType.END_TURN:
          actions.append(idx)
          if action.type == PlayerTaskType.PLAY_CARD and 'Pos' in action.print:
            values[idx] += hand_zone[action.source_position].cost
            is_spell = False
          else:
            if action.type == PlayerTaskType.PLAY_CARD and hand_zone[action.source_position].id in SPELLS.keys():
              values[idx] += hand_zone[action.source_position].cost
              is_spell = True
            else: #action.type == PlayerTaskType.MINION_ATTACK or action.type == PlayerTaskType.HERO_POWER:
              is_spell = False

            if not is_spell:
              # check position
              if action.target_position == 0:
                target = player_hero
              elif action.target_position == 8:
                target = opponent_hero
              elif action.target_position < 8:
                target = player_board[action.target_position - 1]
              elif action.target_position > 8:
                target = opponent_board[action.target_position - 9]
              else:
                raise IndexError

              attacker = player_hero if action.source_position == 0 else player_board[action.source_position - 1]
              atk_dies = attacker.health <= target.atk
              def_dies = target.health <= attacker.atk

              if action.target_position == 8:
                atk_dies = False
                is_hero = True
              else:
                is_hero = False
              if def_dies and is_hero:
                values[idx] += float('inf')
              elif atk_dies and def_dies:
                values[idx] += cost(target) - cost(attacker)
              elif atk_dies and not def_dies:
                values[idx] -= float('inf')
              elif not atk_dies and def_dies:
                overkill = target.health / attacker.atk
                if overkill > 2 and not is_hero:
                  values[idx] = cost(target)
              elif not atk_dies and not def_dies:
                values[idx] += attacker.atk / target.health
            else:
              card_id = hand_zone[action.source_position].id
              if len(opponent_board) == 0 and card_id in [555, 1084, 315, 662] and action.target_position > 8:
                values[idx] = opponent_board[action.target_position - 1].health
              elif len(opponent_board) == 0 and card_id in[1746, 555]:
                values[idx] = 50
              else:
                if len(opponent_board) > 3 and card_id in [564,1004, 447] and action.target_position > 8:
                  values[idx] = opponent_board[action.target_position - 1].health
                elif len(opponent_board) > 0 and opponent_board[action.target_position - 1].atk > 3 and card_id in [77]:
                  values[idx] = opponent_board[action.target_position - 1].health
                else:
                  values[idx] = 0
        else:
          end_turn = idx

      actions = sorted(actions, key=lambda x: values[x], reverse=True)
      if values[actions[0]] < 0:
        selected_action = end_turn
      else:
        selected_action = actions[0]
    return selected_action


class TradingAgent(agents.base_agent.Bot):

  def _choose(self, observation: torch.Tensor, encoded_info: specs.Info):
    # observation, = observation.numpy()
    # player_board = observation[:3 * 7].reshape(-1, 3)
    # opponent_board = observation[3 * 8 + 1:-4].reshape(-1, 3)

    # player_board = tuple(tuple(t) for t in player_board if t.max() > 0)
    # opponent_board = tuple(tuple(t) for t in opponent_board if t.max() > 0)

    # def exit_condition(opponent_hp_left, player_hp_left, node2):
    #   return opponent_hp_left <= 0

    # def prune_condition(opponent_hp_left, player_hp_left, node):
    #   return player_hp_left < 0

    # actions = [
    #   environments.tutorial_environments.make_attack(atk_idx, def_idx, sign=-1)
    #   for atk_idx in range(len(player_board))
    #   for def_idx in range(len(opponent_board))
    # ]
    # for idx in range(len(actions)):
    #   actions[idx].idx = idx  # forgive me guido, for I have sinned

    # _, actions = environments.tutorial_environments.bf_search(actions, exit_condition, prune_condition, opponent_board,
    #                                                           player_board)

    possible_actions = encoded_info['original_info']['possible_actions']
    assert encoded_info['possible_actions'].size(0) == 1

    if len(possible_actions) == 1:
      selected_action = possible_actions[0]
    else:
      actions = []
      values = collections.defaultdict(int)
      end_turn = None
      for action in possible_actions:
        if action.card is not None:
          actions.append(action)

          if action.params['target'].type == CardType.HERO:
            values[action] -= float('inf')

          target = action.params['target']
          attacker = action.card

          atk_dies = attacker.health <= target.atk
          def_dies = target.health <= attacker.atk

          if def_dies:
            values[action] += float('inf')
          elif atk_dies and not def_dies:
            values[action] -= float('inf')
          elif atk_dies and def_dies:
            values[action] += target.cost - attacker.cost
          elif not atk_dies and not def_dies:
            values[action] += attacker.atk / target.health
        else:
          end_turn = action

      actions = sorted(actions, key=lambda x: values[x], reverse=True)
      if values[actions[0]] < 0:
        selected_action = end_turn
      else:
        selected_action = actions[0]

    for enc_action, action_obj in zip(torch.nonzero(encoded_info['possible_actions']),
                                      encoded_info['original_info']['possible_actions']):
      if action_obj == selected_action:
        return enc_action[1]
    else:
      raise Exception
