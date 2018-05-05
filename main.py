#!/usr/bin/env python3.5
import copy
from environment import hearthstone_game
import networks
import collections
import mcts


class Action(object):
    def __init__(self, card, card_object, usage, params, target):
        self.card = card
        self.card_obj = card_object
        self.usage = usage
        self.params = params
        self.vector = [card, target]

    def use(self):
        self.usage(**self.params)

    def __repr__(self):
        return "card: {}, usage: {}, vector: {}".format(self.card_obj, self.params, self.vector)


def play_against_yourself(environment, player_policy):
    pass


def arena_fight(environment, player_policy, opponent_policy):
    pass


def main():
    TRAINING_LEN = 10000
    NUM_ITER = 100
    NUMBER_EPISODES = 100
    UPDATE_THRESHOLD = 0.55

    environment = hearthstone_game.HS_environment()
    player_neural_network = networks.NeuralNetwork(state_size=6, action_size=3*3),
    player_policy = mcts.MCTS(environment, player_neural_network)

    training_samples = collections.deque(maxlen=TRAINING_LEN)

    for iteration_number in range(NUM_ITER):
        for eps in range(NUMBER_EPISODES):
            player_policy.reset()
            new_experiences = play_against_yourself(environment, player_policy)
            training_samples.extend(new_experiences)

        opponent_neural_netwrok = copy.deepcopy(player_neural_network)
        opponent_policy = mcts.MCTS(environment, opponent_neural_netwrok)

        player_neural_network.train(training_samples)

        player_wins, player_losses, draws = arena_fight(environment, player_policy, opponent_policy)

        win_ratio = float(player_wins) / (player_wins + player_losses)

        if win_ratio < UPDATE_THRESHOLD:
            player_neural_network = opponent_neural_netwrok

    # fireplace.cards.db.initialize()
    # # logger = fireplace.logging.getLogger('fireplace')
    # import logging
    # logging.getLogger("fireplace").setLevel(logging.ERROR)
    # player = Agent()
    # sim1 = HSsimulation()
    # cached_lookup_tables = (sim1.possible_states, sim1.possible_actions)
    # games_played = 0
    # games_won = 0
    # games_finished = 0
    # wl_ratio = 0
    # wl_record = []
    # old_miss = 0
    # print("games_played, win %, result, abs_change, turn, q_hit_ratio")
    # while True:
    #     change = 0
    #     abs_change = 0
    #     count = 0
    #     player_actions = 0
    #     player_reward = 0
    #     try:
    #         sim1 = HSsimulation(*cached_lookup_tables)
    #         player.join_game(sim1)
    #         actor_hero = sim1.game.current_player.hero
    #         games_played += 1

    #         while True:
    #             actions = sim1.actions()

    #             if sim1.game.turn > 30:
    #                 sim1.sudden_death()

    #             if len(actions) == 1:
    #                 # end turn
    #                 # sim1.game.end_turn()
    #                 fireplace.utils.play_turn(sim1.game)

    #                 # play opponent turn
    #                 fireplace.utils.play_turn(sim1.game)
    #             else:
    #                 observation = sim1.observe()
    #                 choosen_action = player.choose_action(observation, actions)

    #                 observation, action, next_observation, reward, terminal = sim1.step(choosen_action)
    #                 player_reward += reward
    #                 player_actions += 1

    #                 # print(choosen_action)
    #                 # print(reward)

    #                 delta = player.learn_from_reaction(observation, action, reward, next_observation)
    #                 change += delta
    #                 abs_change += abs(delta)
    #                 count += 1
    #                 if choosen_action.card_obj is None:
    #                     sim1.game.end_turn()

    #                 if terminal:
    #                     break
    #     except GameOver as e:
    #         print_row = False
    #         if sim1.game.turn < 31:
    #             game_result = 1 if sim1.player.playstate == PlayState.WON else 0
    #             games_won += game_result
    #             wl_record.append(game_result)
    #             games_finished += 1
    #             print_row = True

    #             if len(wl_record) > 500:
    #                 del wl_record[0]
    #                 wl_ratio = wl_ratio * 2. / 3. + (sum(wl_record) / len(wl_record)) * 1. / 3.
    #             else:
    #                 print("SKIP")
    #                 wl_ratio = games_won / games_finished

    #         row = "\t".join(map(str, (
    #             0.0000000000000001 + wl_ratio,
    #             sim1.player.playstate,
    #             int(abs_change),
    #             sim1.game.turn,
    #             player.qhit / (player.qmiss + player.qhit),
    #             player.qmiss - old_miss,
    #             0 if player_actions == 0 else player_reward / player_actions,
    #             player_actions,
    #         )))
    #         with open("metrics.tsv", "a") as fout:
    #             fout.write(row + "\n")
    #         """
    #         print(games_played,
    #               0.0000000000000001 + games_won / games_finished,
    #               sim1.player.playstate,
    #               int(abs_change),
    #               sim1.game.turn,
    #              player.qhit / (player.qmiss + player.qhit),
    #              player.qmiss - old_miss,
    #              sep="\t")
    #        """
    #        if print_row:
    #            print(row)
    #        old_miss = player.qmiss


#
#            # print("observation+", observation, "action+", action, "next_observation+", next_observation, "reward+",
#            #  reward)
#            # print("terminal+", terminal)


if __name__ == "__main__":
    main()
