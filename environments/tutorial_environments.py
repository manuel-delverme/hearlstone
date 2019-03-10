import fireplace.cards

import hs_config
import environments.vanilla_hs
import agents.heuristic.hand_coded


class TradingHS(environments.vanilla_hs.VanillaHS):
  def __init__(self, minions_in_board,):
    super(TradingHS, self).__init__(
      max_cards_in_board=minions_in_board,
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
    self.minions_in_board = minions_in_board

  def reinit_game(self, sort_decks=False):
    super(TradingHS, self).reinit_game(sort_decks)
    # wisp = fireplace.cards.filter(name="Wisp")
    raptor = fireplace.cards.filter(name="Bloodfen Raptor", collectible=True)
    wisp = 'CS2_231'
    for _ in range(self.minions_in_board):
      self.simulation.opponent.summon(wisp)
      self.simulation.player.summon(raptor)

  def gather_transition(self):
    game_observation, reward, terminal, info = super(TradingHS, self).gather_transition()
    num_player_minions = len(self.simulation.player.characters[1:])
    num_opponent_minions = len(self.simulation.opponent.characters[1:])
    if num_player_minions == 0 or num_opponent_minions == 0:
      terminal = True
    return game_observation, reward, terminal, info

  def calculate_reward(self):
    num_opponent_minions = len(self.simulation.opponent.characters[1:])
    num_player_minions = len(self.simulation.player.characters[1:])
    if num_opponent_minions > 0:
      return -1
    else:
      return num_player_minions

  def __str__(self):
    return 'TradingHS:{}'.format(
      self.minions_in_board,
    )
