#!/usr/bin/env python3.5
import tqdm
import random
import copy
import environment
import environment.hearthstone_game
import networks
import collections
import mcts
import warnings


def play_one_game_against_yourself(environment, player_policy):
    terminal = False
    state, reward, terminal, info = environment.reset()
    experiences = []
    while not terminal:
        action = player_policy.pick_action(state, info['possible_actions'])
        state, reward, terminal, info = environment.step(action)
        experiences.append((state, reward, terminal))
    return experiences

def arena_fight(environment, player_policy, opponent_policy, nr_games=1000):
    state, possible_actions = environment.reset()
    active_player, passive_player = random.shuffle((player_policy, opponent_policy))
    scoreboard = [0, 0, 0]
    terminal = False

    while not terminal:
        action = active_player.choose_action(state, possible_actions)
        state, reward, terminal, info = environment.step(action)
        possible_actions = info['possible_actions']
        if action == envs.simple_HSenv.GameActions.PASS_TURN or len(possible_actions) == 0:
            active_player, passive_player = passive_player, active_player

    scoreboard[environment.winner] += 1
    win_ratio = float(scoreboard[0]) / sum(scoreboard)
    return win_ratio


def main():
    TRAINING_LEN = 10000
    NUM_ITER = 100
    BATCH_SIZE = 8
    NUMBER_EPISODES = 100
    UPDATE_THRESHOLD = 0.55

    hs_game = environment.hearthstone_game.HS_environment()
    obs, reward, terminal, info = hs_game.reset()

    nr_possible_actions = hs_game.get_nr_possible_actions()

    player_neural_network = networks.NeuralNetwork(
            state_size=len(obs), action_size=nr_possible_actions)

    player_policy = mcts.MCTS(hs_game, player_neural_network)

    training_samples = collections.deque(maxlen=TRAINING_LEN)


    for iteration_number in tqdm.tqdm(range(NUM_ITER), desc='iter'):
        # import ipdb; ipdb.set_trace()
        for eps in tqdm.tqdm(range(NUMBER_EPISODES), 'experience', leave=False):
            player_policy.reset()
            new_experiences = play_one_game_against_yourself(hs_game, player_policy)
            training_samples.extend(new_experiences)

        opponent_neural_netwrok = copy.deepcopy(player_neural_network)
        opponent_policy = mcts.MCTS(hs_game, opponent_neural_netwrok)

        player_neural_network.train(training_samples)
        win_ratio = arena_fight(hs_game, player_policy, opponent_policy)

        if win_ratio < UPDATE_THRESHOLD:
            player_neural_network = opponent_neural_netwrok


if __name__ == "__main__":
    main()
