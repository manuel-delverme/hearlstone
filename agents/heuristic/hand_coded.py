import agents.base_agent
import collections
import random


class HeuristicAgent(agents.base_agent.Agent):
    def __init__(self):
        super().__init__()

    def choose(self, observation, possible_actions):
        if len(possible_actions) == 1:
            return possible_actions[0]

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
            return end_turn
        return actions[0]


class RandomAgent(agents.base_agent.Agent):
    def __init__(self):
        super().__init__()

    def choose(self, observation, possible_actions):
        return random.choice(possible_actions)


def main():
    agent = RandomAgent()
    opponent = HeuristicAgent()

    from environments import simple_env
    from shared import utils
    env = simple_env.SimpleHSEnv(skip_mulligan=True)
    scoreboard = utils.arena_fight(env, agent, opponent, nr_games=1000)
    print('winning ratio', scoreboard)


if __name__ == "__main__":
    main()
