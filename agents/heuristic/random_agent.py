import random
from typing import Dict, Text, Any

import numpy as np

import agents.base_agent


class RandomAgent(agents.base_agent.Bot):
  def _choose(self, observation: np.ndarray, info: Dict[Text, Any]):
    possible_actions = info['possible_actions']
    pa = np.argwhere(possible_actions)
    pa = pa[1, :]
    return random.choice(pa)
