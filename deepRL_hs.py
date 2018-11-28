import agents.base_agent
import agents.heuristic.random_agent
import agents.heuristic.hand_coded
import agents.learning.dqn_agent
import environments.vanilla_hs
import environments.trading_hs
import config
import environments.gym_wrapper


def train():
  hs_game = environments.trading_hs.TradingHS()
  opponent = agents.heuristic.hand_coded.HeuristicAgent()
  hs_game.set_opponent(opponent)

  player = agents.learning.dqn_agent.DQNAgent(
    hs_game.observation_space,
    hs_game.action_space,
    record=not config.enjoy,
  )
  if config.enjoy:
    player.load_model()
    player.render(hs_game)
  else:
    try:
      player.train(hs_game)
    except Exception as e:
      hs_game.dump_log('/tmp/logfile')
      raise e


if __name__ == "__main__":
  train()
