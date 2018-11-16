import gin
from shared import utils
import agents.base_agent

import environments
import environments.trading_hs
import agents.heuristic.random_agent
import agents.heuristic.hand_coded
import agents.learning.dqn_agent


# gin.parse_config_file('config.gin')


# @gin.configurable
def train(
  train_steps=100000,
  test_games=100,
  gamma=0.99,
):
  hs_game = environments.trading_hs.TradingHS()
  try:
    opponent = agents.learning.dqn_agent.DQNAgent(
      hs_game.observation_space,
      hs_game.action_space,
      should_flip_board=True,
      gamma=gamma,
      model_path="checkpoints/checkpointGPU.pth.tar",
    )
    opponent.load_model('checkpoints/opponent.pth.tar')
  except FileNotFoundError:
    opponent = agents.heuristic.hand_coded.HeuristicAgent()
    # opponent = agents.heuristic.random_agent.RandomAgent()
  hs_game.set_opponent(opponent)

  player = agents.learning.dqn_agent.DQNAgent(
    hs_game.observation_space,
    hs_game.action_space,
    gamma=gamma,
  )

  # player.load_model()
  player.train(hs_game, train_steps)

  # scoreboard = utils.arena_fight(hs_game, player, opponent, nr_games=test_games)
  # print(scoreboard)
  # scoreboard = utils.arena_fight(hs_game, opponent, player, nr_games=test_games)
  # print('switched', scoreboard)


if __name__ == "__main__":
  train()
