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
    else:
      raise ValueError
    self.level = level

    super(TradingHS, self).__init__(
      max_cards_in_hand=0,
      skip_mulligan=True,
      starting_hp=hs_config.VanillaHS.starting_hp,
      sort_decks=hs_config.VanillaHS.sort_decks,
    )

    immunity = fireplace.cards.utils.buff(immune=True)
    self.simulation.player.hero.set_current_health(hs_config.VanillaHS.starting_hp)
    self.simulation.player.hero.tags.update(immunity.tags)
    self.simulation.opponent.hero.set_current_health(hs_config.VanillaHS.starting_hp)
    self.simulation.opponent.hero.tags.update(immunity.tags)
    self.opponent = agents.heuristic.hand_coded.PassingAgent()
    self.minions_in_board = level

  def reinit_game(self):
    super(TradingHS, self).reinit_game()

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
    return 'TradingHS:{}'.format(self.level,)
