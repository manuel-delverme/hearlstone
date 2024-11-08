import random

import environments.vanilla_hs
import fireplace.game


def main():
  env = environments.vanilla_hs.VanillaHS(skip_mulligan=True)
  step = 0
  for _ in range(3):
    s, reward, done, possible_actions = env.reset()
    while not done:
      step += 1
      if len(possible_actions) == 1:
        action, = possible_actions
      else:
        for action in possible_actions:
          if action.card is None:
            continue

          card_idx = action.card.zone_position
          zone = action.card.zone
          if zone == fireplace.game.Zone.HAND:
            card_idx += env.simulation._MAX_CARDS_IN_BOARD
        action = random.choice(possible_actions)

      s, r, done, info = env.step(action)
      print(info['stats'])
      # print("step", step, s, r, done, random_act)


if __name__ == '__main__':
  main()
