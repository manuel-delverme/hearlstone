import random
from typing import Dict, Text, Any

import numpy as np

import agents.base_agent


class RandomAgent(agents.base_agent.Bot):
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    pa = info['possible_actions']
    pa = np.argwhere(pa)  # row is always 0
    return random.choice(pa)
