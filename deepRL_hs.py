import agents.base_agent
import agents.heuristic.random_agent
import agents.heuristic.hand_coded
import agents.learning.dqn_agent
import agents.learning.dqn_horizon
import agents.learning.ppo_agent
# import agents.learning.learning_from_heuristics
import environments.vanilla_hs
import environments.trading_hs
import config
import environments.gym_wrapper


def env_loader():
  hs_game = environments.vanilla_hs.VanillaHS()
  hs_game.set_opponent(
    agents.heuristic.hand_coded.HeuristicAgent()
  )
  return hs_game


def train():
  dummy_hs_env = environments.vanilla_hs.VanillaHS()

  # agent = agents.learning.ppo_agent.PPOAgent
  agent = agents.learning.dqn_horizon.DQNAgent
  player = agent(
    dummy_hs_env.observation_space.shape[0],
    dummy_hs_env.action_space.n,
    record=not config.enjoy,
  )
  del dummy_hs_env

  if config.enjoy:
    player.load_model()
    player.render(env_loader())
  else:
    player.load_model()
    player.train(env_loader)


if __name__ == "__main__":
  train()
