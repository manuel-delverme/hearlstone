import agents.base_agent
import agents.heuristic.random_agent
import agents.heuristic.hand_coded
import agents.learning.dqn_agent


def train():
  import environments.trading_hs
  hs_game = environments.trading_hs.TradingHS()
  opponent = agents.heuristic.hand_coded.HeuristicAgent()
  hs_game.set_opponent(opponent)

  player = agents.learning.dqn_agent.DQNAgent(
    hs_game.observation_space,
    hs_game.action_space,
  )
  player.train(hs_game)


if __name__ == "__main__":
  train()
