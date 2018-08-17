#!/usr/bin/env python3.5
from utils import disk_cache
import utils
import string
from collections import defaultdict
import shelve
import pickle
import fireplace
import random

from fireplace import game

import fireplace.logging
from fireplace.game import Game, PlayState, Zone
from fireplace.card import Minion, Secret
from fireplace.exceptions import GameOver
from fireplace.player import Player
from itertools import combinations, product, permutations

import logging

import zerorpc

import gym
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)

class Observations(object):
    PRICE = 0
    INVESTED = 1
    LIQUID = 2


class HS_environment(gym.Env):
    def __init__(self):
        # fireplace.cards.db.initialize()
        import logging
        logging.getLogger("fireplace").setLevel(logging.ERROR)
        self.games_played = 0
        self.games_finished = 0
        self.simulation = HS_client()
        self._seed()

    def reset(self):
        self.games_played += 1
        game_observation, reward, terminal, info = self.simulation.reset()
        return game_observation, reward, False, info

    def play_opponent_turn(self):
        self.simulation.play_random_turn()

    def step(self, action):
        action = utils.to_tuples(action)
        # terminal = False
        action_dict = {k: v for k, v in action}
        game_over = False

        try:
            if action_dict['card'] is None:
                self.simulation.end_turn()
                self.play_opponent_turn()
                o, r, t, info = self.simulation.observe_gamestate()
            else:
                o, r, t, info = self.simulation.step(action)
        except GameOver  as e:
            game_over = True
        except zerorpc.exceptions.RemoteError as e:
            if e.args[0] == 'GameOver':
                game_over = True
            else:
                raise e

        if game_over:
            o, r, t, info = self.simulation.observe_gamestate()

        info['stats'] = self.simulation.get_ascii_board()
        return o, r, t, info

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def HSenv_test():
    env = HS_environment()
    s0, reward, terminal, info = env.reset()
    done = False
    step = 0
    for _ in range(3):
        while not done:
            step += 1
            possible_actions = info['possible_actions']
            random_act = random.choice(possible_actions)
            s, r, done, info = env.step(random_act)
            print(info['stats'])
            print("step", step, r, done)
            input()

class HS_client(object):
    def __init__(self):
        self._client = zerorpc.Client()
        print("connecting..")
        self._client.connect("tcp://127.0.0.1:31337")
        print("done")

    def __getattr__(self, name):
        logger.info("didn't find {}, using RPC".format(name))
        attr = getattr(self._client, name)
        return attr

    class Action(object):
        def __init__(self, card, usage, params, hack):
            self.card = card
            self.usage = usage
            self.params = params
            self.hack = hack

        def use(self):
            self.usage(**self.params)

        def __repr__(self):
            return "card: {}, usage: {}, vector: {}".format(self.card, self.params, None)

        def encode(self):
            state = {'card': None if self.card is None else self.hack.card_to_bow(self.card)}
            params_vec = {}
            for k, v in self.params.items():
                params_vec[k] = self.hack.card_to_bow(v)
            state['params'] = params_vec
            return state

class Agent(object):
    _ACTIONS_DIMENSIONS = 300

    def __init__(self):
        # self.simulation = simulation
        self.epsilon = 0.3
        self.learning_rate = 0.1
        self.gamma = 0.80

        self.qmiss = 1
        self.qhit = 0
        # nr_states = len(simulation.possible_states)
        # nr_actions = len(simulation.possible_actions)
        # self.Q_table = np.zeros(shape=(nr_states, nr_actions))
        self.Q_table = shelve.open("Q_table", protocol=1, writeback=True)
        self.simulation = None

    def join_game(self, simulation: HS_client):
        self.simulation = simulation

    def getQ(self, state_id, action_id=None):
        if action_id == 0:
            return 0
        action_id = str(action_id)
        state_id = str(state_id)
        # state_id = self.simulation.state_to_state_id(state)
        try:
            action_Q = self.Q_table[state_id]
        except KeyError:
            self.Q_table[state_id] = dict()
            action_Q = self.Q_table[state_id]

        if action_id:
            try:
                return action_Q[action_id]
            except KeyError:
                init_q = 0.0
                for nearby_state in range(int(state_id), int(state_id) + 100):
                    try:
                        approximate_q = self.Q_table[str(nearby_state)][action_id]
                        init_q = approximate_q
                        break
                    except KeyError:
                        pass
                self.Q_table[state_id][action_id] = init_q
                return self.Q_table[state_id][action_id]
        else:
            return action_Q

    def setQ(self, state, action, q_value):
        action_id = str(action)
        state_id = str(state)
        self.Q_table[state_id][action_id] = q_value

    def choose_action(self, state, possible_actions):
        # possible_actions = self.simulation.actions()
        if random.random() < self.epsilon:
            best_action = random.choice(possible_actions)
        else:
            q = [self.getQ(state, self.simulation.action_to_action_id(a)) for a in possible_actions]
            maxQ = max(q)
            if maxQ == 0.0:  # FIXME: account for no-action which is always present
                self.qmiss += 1
            else:
                self.qhit += 1
            # choose one of the best actions
            best_action = random.choice([a for q, a in zip(q, possible_actions) if q == maxQ])
        return best_action

    def learn_from_reaction(self, state, action, reward, next_state):
        action_id = self.simulation.action_to_action_id(action)
        state_id = state

        old_q = self.getQ(state_id, action_id)
        change = self.learning_rate * (reward + self.gamma * np.max(self.getQ(next_state)) - old_q)
        new_q = old_q + change
        self.setQ(state, action_id, new_q)
        return change



if __name__ == "__main__":
    HSenv_test()
