import random
import agents.base_agent


class RandomAgent(agents.base_agent.Agent):
  def choose(self, state, possible_actions):
    choice = random.choice(possible_actions)
    return choice

  def __init__(self):
    pass
