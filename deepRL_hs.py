# import agents.base_agent
# import agents.heuristic.hand_coded
# import agents.heuristic.random_agent
# import agents.learning.a2c_agent
# import agents.learning.dqn_agent
# import agents.learning.dqn_horizon
import agents.learning.dqn_simple
# import agents.learning.ppo_agent
# import agents.evolutionary.es
import agents.evolutionary.cmaes
import config
import environments.gym_wrapper
import environments.trading_hs
import environments.vanilla_hs


def make_env(seed=None, env_id=None, log_dir=None, episode_life=None):
  hs_game = environments.vanilla_hs.VanillaHS()
  opponent = config.VanillaHS.opponent()
  hs_game.set_opponent(opponent)
  return hs_game


def train() -> None:
  dummy_hs_env = environments.vanilla_hs.VanillaHS()

  agent = agents.learning.dqn_simple.DQNAgent
  player = agent(
    dummy_hs_env.observation_space.shape[0],
    dummy_hs_env.action_space.n,
    record=not config.enjoy,
  )
  del dummy_hs_env

  # player.load_model()
  if config.enjoy:
    # player.render(make_env())
    player.train(make_env)
  else:
    player.train(make_env)


if __name__ == "__main__":
  train()
