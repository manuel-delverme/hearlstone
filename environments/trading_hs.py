import collections
import random

import gym
import numpy as np
import tqdm

import environments.base_env
import environments.sabber_hs
import hs_config

P_HP = 0
P_ATK = 1
P_EXAUSTED = 2
PLAYER = slice(P_HP, P_EXAUSTED)
O_HP = 3
O_ATK = 4
OPPONENT = slice(O_HP, O_ATK)
RELATIVE_ATK, RELATIVE_HP = 0, 1


def make_attack(atk_idx, def_idx, sign=-1):
  def attack(node):
    player_board, opponent_board = node

    if atk_idx == len(player_board):
      player_board = (*player_board, (random.randint(1, 9), 0))

    if def_idx == len(opponent_board):
      opponent_board = (*opponent_board, (random.randint(1, 9), 0))

    attacker = list(player_board[atk_idx])
    defender = list(opponent_board[def_idx])

    attacker[RELATIVE_HP] += sign * defender[RELATIVE_ATK]
    if 0 > attacker[RELATIVE_HP] or attacker[RELATIVE_HP] > 9:
      raise ValueError

    defender[RELATIVE_HP] += sign * attacker[RELATIVE_ATK]
    if 0 > defender[RELATIVE_HP] or defender[RELATIVE_HP] > 9:
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
  node2 = None

  closed = set()

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
        continue
      except ValueError:
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


class TradingHS(environments.base_env.RenderableEnv):
  id_to_action = [environments.base_env.BaseEnv.GameActions.PASS_TURN, ] + [
    (s, t) for s in range(hs_config.Environment.max_cards_in_board) for t in range(hs_config.Environment.max_cards_in_board)]
  action_to_id = {v: k for k, v in enumerate(id_to_action)}
  board = np.zeros((5, hs_config.Environment.max_cards_in_board), dtype=np.int8)  # player (hp, atk, exaust), opponent (hp, atk)

  action_space = gym.spaces.Discrete(n=len(id_to_action))
  observation_space = gym.spaces.Box(low=-1, high=100, shape=(len(board.flatten()),), dtype=np.int)

  def __init__(self, *, address: str = None, seed: int = None, env_number: int = None):
    super().__init__()
    self.board.fill(0)
    self.turns = 0

  def agent_game_vale(self):
    if self.turns > 0:
      reward = -1
    else:
      reward = np.all(self.board[OPPONENT] <= 0)
    return np.float32(reward)

  def step(self, action_id: np.ndarray):
    action_id = int(action_id)
    assert self.last_info['possible_actions'][action_id]
    if self.id_to_action[action_id] == environments.base_env.BaseEnv.GameActions.PASS_TURN:
      self.board[P_EXAUSTED, :].fill(0)
      self.turns += 1
    else:
      attacker, defender = self.id_to_action[action_id]

      assert self.board[P_ATK, attacker] > 0
      assert self.board[O_ATK, defender] > 0

      self.board[O_HP, defender] -= self.board[P_ATK, attacker]
      self.board[P_HP, attacker] -= self.board[O_ATK, defender]

      if self.board[O_HP, defender] < 1:
        self.board[OPPONENT, defender].fill(0)

      if self.board[P_HP, attacker] < 1:
        self.board[PLAYER, attacker].fill(0)

    return self.gather_transition()

  def render(self, **kwargs):
    print("=" * 4, "OPPONENT", "=" * 4)
    print(self.board[O_HP, :], "HP")
    print(self.board[O_ATK, :], "ATK")
    print("=" * 18)
    print(self.board[P_ATK, :], "ATK")
    print(self.board[P_HP, :], "HP")
    print("=" * 5, "PLAYER", "=" * 5)

  def reset(self):
    self.turns = 0
    self.board.fill(0)
    player_board, opponent_board = self.generate_board(acceptable_minions=(4, 5))
    for minion_board_position, minion in enumerate(player_board):
      self.board[P_HP: P_ATK + 1, minion_board_position] = minion

    for minion_board_position, minion in enumerate(opponent_board):
      self.board[O_HP: O_ATK + 1, minion_board_position] = minion

    return self.gather_transition()

  def generate_board(self, acceptable_minions):
    def exit_condition(opponent_hp_left, player_hp_left, node):
      return opponent_hp_left > 0 and player_hp_left > 0 and len(node[0]) + len(node[1]) in acceptable_minions

    def prune_condition(opponent_hp_left, player_hp_left, node):
      return len(node[0]) + len(node[1]) > max(acceptable_minions)

    actions = [
      make_attack(atk_idx, def_idx, sign=+1)
      for atk_idx in range(hs_config.Environment.max_cards_in_board)
      for def_idx in range(hs_config.Environment.max_cards_in_board)
    ]
    player_board, opponent_board = tuple(), tuple()

    (player_board, opponent_board), _ = bf_search(actions, exit_condition, prune_condition, player_board, opponent_board)

    player_board = (*player_board, (random.randint(1, 3), random.randint(1, 3)))
    return player_board, opponent_board

  def gather_transition(self):
    reward = self.agent_game_vale()
    terminal = bool(reward)
    observation = self.board.flatten()
    info = {'possible_actions': self.gather_possible_actions(), }

    if terminal:
      info['game_statistics'] = {
        'outcome': reward,
        'opponent_nr': -1,
        'episode_empowerment': float('nan')
      }

    self.last_info = info
    return observation, reward, terminal, info

  def gather_possible_actions(self):
    possible_actions = np.zeros(hs_config.Environment.max_cards_in_board * hs_config.Environment.max_cards_in_board + 1, dtype=np.float32)

    for action_idx, action in enumerate(self.id_to_action):
      if action == environments.base_env.BaseEnv.GameActions.PASS_TURN:
        possible_actions[action_idx] = 1
      else:
        attacker_pos, defender_pos = action
        if (
            self.board[P_HP, attacker_pos] > 0 and
            self.board[P_ATK, attacker_pos] > 0 and
            not self.board[P_EXAUSTED, attacker_pos] and
            self.board[O_HP, defender_pos] > 0
        ):
          possible_actions[action_idx] = 1
    return possible_actions


if __name__ == "__main__":
  env = TradingHS()
  observation, reward, done, info = env.reset()
  r_max = 0
  for _ in tqdm.tqdm(range(10000)):
    env.render()
    pa = info['possible_actions']
    pa_list = np.argwhere(pa).squeeze(-1)
    a = np.random.choice(pa_list)
    _, r, done, info = env.step(a)
    r_max = max(r, r_max)
    if done:
      _, _, done, info = env.reset()
      print("=" * 10, "NEW GAME", "=" * 10, r_max)
