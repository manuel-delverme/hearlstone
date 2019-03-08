import agents.base_agent
import numpy as np
import random


class RandomAgent(agents.base_agent.Agent):
  def __init__(self):
    super().__init__()

  def choose(self, observation: np.array, info: dict):
    possible_actions = info['possible_actions']
    choice = random.choice(possible_actions)
    return choice
