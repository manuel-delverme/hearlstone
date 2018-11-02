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
    hs_game.action_space,
    gamma=0.99
  )
  player.train(
    hs_game,
    num_frames=100000,
    eval_every=2000
  )
  scoreboard = utils.arena_fight(hs_game, player, opponent, nr_games=100)
  print(scoreboard)


if __name__ == "__main__":
  main()
