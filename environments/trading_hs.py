from environments import simple_hs
from typing import Tuple, List
from agents.heuristic import hand_coded
from environments import simulator


class TradingHS(simple_hs.VanillaHS):
  def __init__(self):
    self.minion_player_agent = hand_coded.HeuristicAgent()
    super().__init__(
      skip_mulligan=True,
      cheating_opponent=True,
      max_cards_in_game=7,
    )

  def reset(self):
    super().reset()
    observation, _, _, info = self.gather_transition()
    self.fast_forward_game(observation, info)
    return self.gather_transition()

  def step(self, action: Tuple[int, int]):
    observation, reward, terminal, info = super(TradingHS, self).step(action)
    if not terminal:
      self.fast_forward_game(observation, info)
    return self.gather_transition()

  def fast_forward_game(self, o, info):
    non_trade_actions_obj, non_trade_actions_enc =self.gather_play_from_hand_acts(info)
    if len(non_trade_actions_obj) == 0:
      return

    restricted_info = info.copy()
    # restrict the agent to non-trade actions
    restricted_info['original_info']['possible_actions'] = non_trade_actions_obj
    restricted_info['possible_actions'] = non_trade_actions_enc

    action = self.minion_player_agent.choose(o, restricted_info)
    o, r, t, info = super(TradingHS, self).step(action)
    assert r == 0.0
    return self.gather_transition()

  @staticmethod
  def gather_play_from_hand_acts(info: dict):
    assert isinstance(info['original_info']['possible_actions'][0], simulator.HSsimulation.Action)
    assert isinstance(info['possible_actions'][0], tuple)

    acts_obj = info['original_info']['possible_actions']
    acts_enc = info['possible_actions']

    non_trade_actions_obj = []
    non_trade_actions_enc = []
    for act_enc, act_obj in zip(acts_enc, acts_obj):
      if act_obj.card is None:
        continue
      if act_obj.params['target'] is None:
        non_trade_actions_obj.append(act_obj)
        non_trade_actions_enc.append(act_enc)
    return non_trade_actions_obj, non_trade_actions_enc

  def play_opponent_turn(self):
    while self.simulation.game.current_player.controller.name == 'Opponent':
      self.play_opponent_action()

  def play_opponent_action(self):
    observation, _, _, info = self.gather_transition()
    self.fast_forward_game(observation, info)

    observation, _, terminal, info = self.gather_transition()

    action = self.opponent.choose(observation, info)
    self.step(action)

    self.fast_forward_game(observation, info)

    trans = self.gather_transition()
    return trans

