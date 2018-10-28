from abc import ABC

import gym


class BaseEnv(gym.Env, ABC):
    class GameActions(object):
        PASS_TURN = 0

    def play_opponent_turn(self):
        raise NotImplemented

    def game_value(self):
        raise NotImplemented
