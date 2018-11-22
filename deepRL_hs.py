import agents.base_agent
import agents.heuristic.random_agent
import agents.heuristic.hand_coded
import agents.learning.dqn_agent


def train():
  import environments.trading_hs
  hs_game = environments.trading_hs.TradingHS()
  opponent = agents.heuristic.hand_coded.HeuristicAgent()
  hs_game.set_opponent(opponent)

  # player = agents.learning.dqn_agent.DQNAgentBaselines(
  player = agents.learning.dqn_agent.DQNAgent(
    hs_game.observation_space,
    hs_game.action_space,
  )
  import environments.gym_wrapper
  hs_game = environments.gym_wrapper.GymWrapper(hs_game)

  # player.load_model()
  player.train(hs_game)

  # scoreboard = utils.arena_fight(hs_game, player, opponent, nr_games=test_games)
  # print(scoreboard)
  # scoreboard = utils.arena_fight(hs_game, opponent, player, nr_games=test_games)
  # print('switched', scoreboard)


if __name__ == "__main__":
  train()
