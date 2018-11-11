import gin
from shared import utils
import agents.base_agent

import environments
import environments.simple_hs
import agents.heuristic.random_agent
import agents.heuristic.hand_coded
import agents.learning.dqn


@gin.configurable
def main():
  # opponent = agents.heuristic.random_agent.RandomAgent()
  hs_game = environments.simple_hs.TradingHS()
  try:
    opponent = agents.learning.dqn.DQNAgent(
      hs_game.observation_space,
      hs_game.action_space,
      gamma=0.99
    )
    opponent.load_model('checkpoints/opponent.pth.tar')
  except FileNotFoundError:
    opponent = agents.heuristic.hand_coded.HeuristicAgent()

  hs_game.set_opponent(None)

  player = agents.learning.dqn.DQNAgent(
    hs_game.observation_space,
    hs_game.action_space,
    gamma=0.99
  )

  player.train(
    hs_game,
    opponent=opponent,
    num_frames=8000,
    eval_every=800
  )
  scoreboard = utils.arena_fight(hs_game, player, opponent, nr_games=100)
  print(scoreboard)


if __name__ == "__main__":
  main()
