import collections
import random
from typing import Dict, Text, Any

import numpy as np
import torch

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
  def __init__(self, level: int = hs_config.VanillaHS.level):
    assert -1 < level < 6
    super().__init__()
    self.randomness = [
      None,
      1,
      0.75,
      0.5,
      0.25,
      0.125,
      0,
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
