import gin
from shared import utils
import agents.base_agent

import environments
import environments.simple_hearthstone_game
import agents.heuristic.random_agent


@gin.configurable
def main():
  opponent = agents.heuristic.random_agent.RandomAgent()
  player = train_dqn()

  stats = test_player(nr_games=100, opponent=opponent, player=player)
  print(stats)


def test_player(
    nr_games: int,
    player: agents.base_agent.Agent,
    opponent: agents.base_agent.Agent
):
  hs_game = environments.simple_hearthstone_game.VanillaHS(
      skip_mulligan=True,
      max_cards_in_game=1,
  )

  for game_num in range(nr_games):

    utils.arena_fight(
        environment=hs_game,
        player_policy=player,
        opponent_policy=opponent,
        nr_games=100,
    )

    obs, reward, terminal, info = hs_game.reset()
    done = False
    nr_turns = 0
    while not done:
      nr_turns += 1

      player_action = None
      while player_action != environments.simple_hearthstone_game.GameActions.PASS_TURN:
        possible_actions = info['possible_actions']
        player_action = player.step(possible_actions)
        s, r, done, info = hs_game.step(player_action)

      opponent_action = None
      while opponent_action != environments.simple_hearthstone_game.GameActions.PASS_TURN:
        possible_actions = info['possible_actions']
        opponent_action = opponent.step(possible_actions)
        s, r, done, info = hs_game.step(opponent_action)

        # TODO: check winner
      print(hs_game.render(mode='ASCII'))
  return winning_ratio

def train_dqn():
  return agents.heuristic.random_agent.RandomAgent()


if __name__ == "__main__":
  main()
