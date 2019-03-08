import agents.learning.dqn_simple
import baselines
import agents.evolutionary.cmaes
import agents.search.classical

import hs_config
import environments.gym_wrapper
import environments.trading_hs
import environments.vanilla_hs


def train(level) -> None:
  def make_env(seed=None, env_id=None, log_dir=None, episode_life=None):
    hs_game = environments.vanilla_hs.VanillaHS()
    opponent = hs_config.VanillaHS.opponent(level)
    hs_game.set_opponent(opponent)
    return hs_game

  dummy_hs_env = environments.vanilla_hs.VanillaHS()

  agent = agents.learning.dqn_simple.DQNAgent
  player = agent(
    dummy_hs_env.observation_space.shape[0],
    dummy_hs_env.action_space.n,
    record=not hs_config.enjoy,
    experiment_name='lvl:{}_'.format(level),
  )
  del dummy_hs_env

  # player.load_model()
  if hs_config.enjoy:
    # player.render(make_env())
    player.train(make_env)
  else:
    player.train(make_env)


if __name__ == "__main__":
  train(0)
