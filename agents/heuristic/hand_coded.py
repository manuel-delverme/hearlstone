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
