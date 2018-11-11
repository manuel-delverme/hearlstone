import agents.base_agent
import numpy as np
import collections


class HeuristicAgent(agents.base_agent.Agent):
  def __init__(self):
    super().__init__()

  def choose(self, observation: np.array, info: dict):
    possible_actions = info['original_info']['possible_actions']

    if len(possible_actions) == 1:
      selected_action = possible_actions[0]
    else:
      actions = []
      values = collections.defaultdict(int)
      end_turn = None
      for action in possible_actions:
        if action.card is not None:
          actions.append(action)
          if 'target' not in action.params or action.params['target'] is None:
            # play card
            values[action] += action.card.cost
          else:
            target = action.params['target']
            attacker = action.card

            atk_dies = attacker.health <= target.atk
            def_dies = target.health <= attacker.atk
            if 'HERO' in target.id:
              atk_dies = False
              is_hero = True
            else:
              is_hero = False

            if def_dies and is_hero:
              values[action] += float('inf')
            elif atk_dies and def_dies:
              values[action] += target.cost - attacker.cost
            elif atk_dies and not def_dies:
              values[action] -= float('inf')
            elif not atk_dies and def_dies:
              overkill = target.health / attacker.atk
              if overkill > 2 and not is_hero:
                values[action] = target.cost
            elif not atk_dies and not def_dies:
              values[action] += attacker.atk / target.health
        else:
          end_turn = action

      actions = sorted(actions, key=lambda x: values[x], reverse=True)
      if values[actions[0]] < 0:
        selected_action = end_turn
      else:
        selected_action = actions[0]

    for enc_action, action_obj in zip(info['possible_actions'], info['original_info']['possible_actions']):
      if action_obj == selected_action:
        return enc_action
    else:
      raise Exception


def main():
  import agents.heuristic.random_agent
  agent = agents.heuristic.random_agent()
  opponent = HeuristicAgent()

  from environments import simple_hs
  from shared import utils
  env = simple_hs.VanillaHS()
  scoreboard = utils.arena_fight(env, agent, opponent, nr_games=1000)
  print('winning ratio', scoreboard)


if __name__ == "__main__":
  main()
