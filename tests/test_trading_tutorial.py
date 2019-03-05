import fireplace.cards
import numpy as np

import environments.tutorial_environments

raptor = fireplace.cards.filter(name="Bloodfen Raptor", collectible=True)


def test_level0():
  env = environments.tutorial_environments.TradingHS()
  o, _, _, i = env.reset()
  assert np.all(o[0:3] == [3, 2, 0])
  assert o[3:15].max() == -1
  # assert np.all(o[15:18] == [0, 30, 0])
  assert o[18] == 1  # player mana
  assert np.all(o[19:22] == [1, 1, 0])
  assert o[22:-4].max() == -1
  # assert np.all(o[-4:-1] == [0, 30, 0])
  assert o[-1] == 0  # opponent mana
  o, r, t, i = env.step(0)
  assert r == -1

  assert np.all(o[0:3] == [3, 2, 1])
  assert o[3:15].max() == -1
  # assert np.all(o[15:18] == [0, 30, 0])
  assert o[18] == 2  # player mana
  assert np.all(o[19:22] == [1, 1, 0])
  assert o[22:-4].max() == -1
  # assert np.all(o[-4:-1] == [0, 30, 0])
  assert o[-1] == 1  # opponent mana
  o, r, t, i = env.step(2)
  assert r == 1
  assert t is True


test_level0()
