import random
from typing import Dict, Text, Any

import numpy as np

import agents.base_agent


class RandomAgent(agents.base_agent.Bot):
  def __init__(self):
    super().__init__()

  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    possible_actions = info['possible_actions']
    choice = random.choice(possible_actions)
    return choice
