import collections
import random
from typing import Dict, Text, Any

import numpy as np
import torch

import agents.base_agent
import agents.heuristic.random_agent
import environments.sabber_hs
import hs_config
import shared.constants as C
import shared.utils
import specs


def minion_value(m):
  return m.atk / 2 + m.health / 2.


def parse_hero(hero):
  return hero.atk, hero.base_health - hero.damage, hero.exhausted, hero.power.exhausted


class PassingAgent(agents.base_agent.Agent):
  def __init__(self):
    super().__init__()

  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    return 0


class HeuristicAgent(agents.base_agent.Bot):
  def __init__(self):
    self.logger = shared.utils.HSLogger(__name__, log_to_stdout=hs_config.log_to_stdout)
    super().__init__()
    self.random_agent = agents.heuristic.random_agent.RandomAgent()
    self.passing_agent = PassingAgent()

  def _choose(self, observation: np.ndarray, encoded_info: specs.Info):
    encoded_info['possible_actions'] = torch.FloatTensor(encoded_info['possible_actions']).unsqueeze(0)
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
  game_snapshot = info['game_snapshot']

  player_hero = C.Hero(*parse_hero(game_snapshot.CurrentPlayer.hero))
  opponent_hero = C.Hero(*parse_hero(game_snapshot.CurrentOpponent.hero))

  hand_zone = list(map(lambda x: C.Card(*environments.sabber_hs.parse_card(x)), game_snapshot.CurrentPlayer.hand_zone.entities))
  player_board = list(map(lambda x: C.Minion(*environments.sabber_hs.parse_minion(x)), game_snapshot.CurrentPlayer.board_zone.minions))
  opponent_board = list(map(lambda x: C.Minion(*environments.sabber_hs.parse_minion(x)), game_snapshot.CurrentOpponent.board_zone.minions))
  return player_hero, opponent_hero, hand_zone, player_board, opponent_board


class SabberAgent(HeuristicAgent):
  def _choose(self, observation: np.ndarray, encoded_info: specs.Info):
    possible_actions = encoded_info['original_info']['game_options']
    player_hero, opponent_hero, hand_zone, player_board, opponent_board = parse_game(encoded_info['original_info'])
    if hs_config.Environment.ENV_DEBUG_HEURISTIC:
      desk = {}

    if len(possible_actions) == 1:
      selected_action = 0
    else:
      actions = []
      values = collections.defaultdict(int)
      end_turn = None
      for idx, action in possible_actions.items():
        if action.type == C.PlayerTaskType.END_TURN:
          end_turn = idx
          continue

        actions.append(idx)

        if self.action_is_play_minion(action):
          minion = hand_zone[action.source_position]
          value = minion.cost
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            desk[idx] = f"{action.print}[PLAY_MINION] has value of {value}"

        elif self.action_is_spell(action, hand_zone):
          value = self.evaluate_spell(action, hand_zone, opponent_board, opponent_hero)
          try:
            target_entity = player_board[action.target_position] if action.target_position < 9 else opponent_board[
              action.target_position]
          except Exception:
            target_entity = None
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            desk[idx] = f"{action.print}[SPELL] on {target_entity} has value of {value}"

        elif self.action_is_trade(action, hand_zone):
          value = self.evaluate_trade(action, opponent_board, opponent_hero, player_board, player_hero)
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            desk[idx] = f"{action.print}[TRADE] has value of {value}"

        elif self.action_is_hero_power(action, hand_zone):
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            desk[idx] = f"{action.print}[TRADE] SKIPPED"

          value = self.evaluate_hero_power(action, hand_zone, opponent_board, opponent_hero)
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            desk[idx] = f"{action.print}[TRADE] has value of {value}"
        else:
          raise NotImplementedError

        values[idx] = value

      actions = sorted(actions, key=lambda x: values[x], reverse=True)
      if hs_config.Environment.ENV_DEBUG_HEURISTIC:
        for idx in actions:
          self.logger.debug(desk[idx])  # value[idx]

      if values[actions[0]] < 0:
        if hs_config.Environment.ENV_DEBUG:
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            self.logger.debug('Passing')
        selected_action = end_turn
      else:
        selected_action = actions[0]
        if hs_config.Environment.ENV_DEBUG_HEURISTIC:
          if hs_config.Environment.ENV_DEBUG_HEURISTIC:
            self.logger.debug("".join(['Playing', desk[selected_action]]))
    return selected_action

  def action_is_play_minion(self, action):
    return action.type == C.PlayerTaskType.PLAY_CARD and 'Pos' in action.print

  def action_is_trade(self, action, hand_zone):
    return action.type == C.PlayerTaskType.MINION_ATTACK

  def action_is_hero_power(self, action, hand_zone):
    return action.type == C.PlayerTaskType.HERO_POWER

  def action_is_hero_attack(self, action, hand_zone):
    return action.type == C.PlayerTaskType.HERO_ATTACK

  def action_is_spell(self, action, hand_zone):
    return action.type == C.PlayerTaskType.PLAY_CARD and hand_zone[action.source_position].id in C.SPELL_IDS

  def evaluate_trade(self, action, opponent_board, opponent_hero, player_board, player_hero):
    value = 0
    # check position
    if action.target_position == 0:
      target = player_hero
      is_hero = True
    elif action.target_position == 8:
      target = opponent_hero
      is_hero = True
    elif action.target_position < 8:
      target = player_board[action.target_position - 1]
      is_hero = False
    elif action.target_position > 8:
      target = opponent_board[action.target_position - 9]
      is_hero = False
    else:
      raise IndexError

    attacker = player_hero if action.source_position == 0 else player_board[action.source_position - 1]
    atk_dies = attacker.health <= target.atk
    def_dies = target.health <= attacker.atk
    if action.target_position in (0, 8):
      atk_dies = False

    if def_dies and is_hero:
      value += float('inf')
    elif atk_dies and def_dies:
      value += minion_value(target) - minion_value(attacker)
    elif atk_dies and not def_dies:
      value -= float('inf')
    elif not atk_dies and def_dies:
      overkill = target.health / attacker.atk
      if overkill > 2 and not is_hero:
        value = minion_value(target)
    elif not atk_dies and not def_dies:
      value += attacker.atk / target.health
    return value

  def evaluate_hero_power(self, action, hand_zone, opponent_board, opponent_hero):
    if action.target_position < 8:
      return -1  # hitting your own minions as mage is not a good idea

    if action.target_position == -1:
      target_minion = None
    elif action.target_position < 8:
      target_minion = opponent_board[action.target_position - 1]
    elif action.target_position == 8:
      target_minion = opponent_hero
    else:
      target_minion = opponent_board[action.target_position - 9]

    if action.target_position == 8 and opponent_hero.health <= 1:  # kill him!
      value = float('inf')
    elif target_minion.health in (1,):  # kill minion
      value = target_minion.atk
    else:
      value = 0.001
    return value

  def evaluate_spell(self, action, hand_zone, opponent_board, opponent_hero):
    card_id = hand_zone[action.source_position].id
    if action.target_position < 8:
      return -1  # hitting your own minions as mage is not a good idea

    if action.target_position == -1:
      target_minion = None
    elif action.target_position < 8:
      target_minion = opponent_board[action.target_position - 1]
    elif action.target_position == 8:
      target_minion = opponent_hero
    else:
      target_minion = opponent_board[action.target_position - 9]

    value = -1
    if card_id == C.SPELLS.ArcaneIntellect:
      if len(hand_zone) < 4:
        value = 0.01
    elif card_id == C.SPELLS.MirrorImage:
      for target_minion in opponent_board:
        if target_minion.atk > 6:  # slow down big minions
          value += 0.2 * target_minion.atk
    elif card_id == C.SPELLS.Fireball:
      if action.target_position == 8 and opponent_hero.health <= 6:  # kill him!
        value = float('inf')
      elif target_minion.health in (4, 5, 6):  # kill minion
        value = target_minion.atk
      else:  # slow down big minions
        value = -1
    elif card_id == C.SPELLS.Frostbolt:
      if action.target_position == 8 and opponent_hero.health <= 3:  # kill him!
        value = float('inf')
      elif target_minion.health in (2, 3):  # kill minion
        value = target_minion.atk
      else:  # slow down big minions
        value = 1 - 0.2 * target_minion.atk
    elif card_id == C.SPELLS.Polymorph:
      value = minion_value(target_minion) - 1
    elif card_id == C.SPELLS.ArcaneExplosion:
      for target_minion in opponent_board:
        if target_minion.health == 1:  # slow down big minions
          value += target_minion.atk
    elif card_id == C.SPELLS.FrostNova:
      if len(opponent_board) > 2:
        for target_minion in opponent_board:
          value += 0.2 * minion_value(target_minion)
    elif card_id == C.SPELLS.ArcaneMissels:
      if opponent_hero.health < 2:
        value = float('inf')
      else:
        for target_minion in opponent_board:
          if target_minion.health == 1:  # slow down big minions
            value += 3. / len(opponent_board)
    elif card_id == C.SPELLS.Flamestrike:
      for target_minion in opponent_board:
        if target_minion.health <= 4:  # slow down big minions
          value += minion_value(target_minion)
    elif card_id == C.SPELLS.TheCoin:
      value = random.choice((float('inf'), -1))
    return value
