import enum

import gym.spaces
import numpy as np

import environments.base_env
import environments.sabber_hs
import hs_config
from shared import constants as C


class Board(enum.IntEnum):
  HP = 0
  ATK = 1
  EXAUSTED = 2


class TradingHS(environments.base_env.RenderableEnv):
  actions = [(s, t) for s in range(hs_config.Environment.max_cards_in_board) for t in range(hs_config.Environment.max_cards_in_board)]
  action_to_id = {v: k for k, v in enumerate(actions)}
  board_to_board = {C.PlayerTaskType.MINION_ATTACK}

  def __init__(self, *, address: str = None, seed: int = None, env_number: int = None):
    super().__init__()
    self.action_space = gym.spaces.Discrete(n=len(self.action_to_id))
    self.observation_space = gym.spaces.Box(low=-1, high=100, shape=(C.STATE_SPACE,), dtype=np.int)
    self.reset()

  def agent_game_vale(self):
    return np.float32(np.all(self.opponent_board == 0))

  def step(self, action_id: np.ndarray):
    source, target = self.actions[action_id]
    reward = self.agent_game_vale()
    terminal = not reward
    observation = np.stack((self.player_board, self.opponent_board))
    info = {
      'possible actions': np.ones_like()
    }
    # self.last_info = info
    # self.last_observation = observation
    return observation, reward, terminal, info

  def reset(self):
    self.opponent_board = np.zeros((3, hs_config.Environment.max_cards_in_board))
    self.player_board = np.zeros((3, hs_config.Environment.max_cards_in_board))
    observation, possible_actions = self.parse_game()
    info = {'possible_actions': possible_actions, }
    self.last_info = info
    self.last_observation = observation
    return observation, 0, False, info

  def parse_game(self):
    obs = np.array((
      *self.player_board,
      *self.opponent_board,
    ), dtype=np.int32)
    pa = self.gather_possible_actions()
    assert obs.shape[0] == C.STATE_SPACE
    assert pa.shape[0] == C.ACTION_SPACE
    return obs, pa

  def gather_possible_actions(self):
    pass
