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

class HS_environment(gym.Env):
    """
    wrapper around the client, this handles anything which is multi game stats and ratios if needded
    """
    def __init__(self):
        self.games_played = 0
        self.games_finished = 0
        self.simulation = HS_client()

    def reset(self):
        self.games_played += 1
        game_observation, reward, terminal, info = self.simulation.reset()
        return game_observation, reward, False, info

    def get_nr_possible_actions(self):
        return self.simulation.get_nr_possible_actions()

    def step(self, action):
        # action_dict = {k: v for k, v in action}
        game_over = False

        try:
            if action == 'pass':
                self.simulation.end_turn()
                # TODO: make opponent play with policy
                # opponent plays random for now
                self.simulation.play_random_turn()
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

class HS_client(object):
    """
        dumb proxy for RPC server
    """

    def __init__(self):
        self._client = zerorpc.Client()
        print("connecting..")
        self._client.connect("tcp://127.0.0.1:31337")
        print("done")

    def __getattr__(self, name):
        logger.info("didn't find {}, using RPC".format(name))
        attr = getattr(self._client, name)
        return attr


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


if __name__ == "__main__":
    HSenv_test()
