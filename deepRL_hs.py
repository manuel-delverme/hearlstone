import gin
from shared import utils
import agents.base_agent

import environments
import environments.simple_hearthstone_game
import agents.heuristic.random_agent
import agents.learning.dqn


@gin.configurable
def main():
  hs_game = environments.simple_hearthstone_game.TradingHS()
  opponent = agents.heuristic.random_agent.RandomAgent()

  player = agents.learning.dqn.DQNAgent(
    hs_game.observation_space,
    hs_game.observation_space,
    gamma=0.99
  )

  scoreboard = utils.arena_fight(hs_game, player, opponent, nr_games=1000)
  print(scoreboard)


def train_dqn():
  return agents.heuristic.random_agent.RandomAgent()

if __name__ == "__main__":
  main()
