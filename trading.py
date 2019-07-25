from typing import Callable, Type

import agents.base_agent
import hs_config
import main
from environments import base_env

if __name__ == "__main__":
  def get_game_mode() -> Callable[[], base_env.BaseEnv]:
    import environments.tutorial_environments
    return environments.tutorial_environments.TradingHS


  def get_opponent() -> Type[agents.base_agent.Agent]:
    import agents.heuristic.hand_coded
    return agents.heuristic.hand_coded.TradingAgent


  hs_config.Environment.get_game_mode = get_game_mode
  hs_config.Environment.get_opponent = get_opponent

  main.train()
