#!/usr/bin/env python3.5
import shelve
import pickle
import fireplace
import random

from fireplace import game
from hearthstone.enums import CardClass, CardType
import numpy as np

import fireplace.logging
from fireplace.game import Game, PlayState, Zone
from fireplace.card import Minion, Secret
from fireplace.exceptions import GameOver
from fireplace.player import Player
from itertools import combinations, product, permutations

import gym
from utils import disk_cache
from gym import spaces
from gym.utils import seeding


class Action(object):
    def __init__(self, card, usage, params):
        self.card = card
        self.usage = usage
        self.params = params

    def use(self):
        self.usage(**self.params)

    def __repr__(self):
        return "card: {}, usage: {}, vector: {}".format(self.card, self.params, None)


class Observations(object):
    PRICE = 0
    INVESTED = 1
    LIQUID = 2


class HSEnv(gym.Env):
    def __init__(self):
        fireplace.cards.db.initialize()
        import logging
        logging.getLogger("fireplace").setLevel(logging.ERROR)
        self.sim1 = HSsimulation()
        self.games_played = 0
        self.games_finished = 0
        self.simulation = HSsimulation()
        self._seed()

    def _reset(self):
        self.actor_hero = self.sim1.game.current_player.hero
        self.games_played += 1
        possible_actions = self.sim1.actions()
        game_observation = self.sim1.observe()
        reward = self.sim1.reward()

        return game_observation, reward, False, {
            'possible_actions': possible_actions
        }

    def play_opponent_turn(self):
        fireplace.utils.play_turn(self.sim1.game)

    def _step(self, action):
        terminal = False

        if action.card is None:
            self.sim1.game.end_turn()
            try:
                self.play_opponent_turn()
            except GameOver as e:
                terminal = True
        else:
            try:
                observation, reward, terminal = self.sim1.step(action)
            except GameOver as e:
                terminal = True

        possible_actions = self.sim1.actions()
        game_observation = self.sim1.observe()
        reward = self.sim1.reward()
        stats = ""
        stats += "-" * 100
        stats += "\nYOU:{player_hp}\n{player_hand}\n{player_board}\nboard:{o_board}\nENEMY:{o_hp}\thand:{o_hand}".format(
            player_hp=self.sim1.player.hero.health,
            player_hand="\t".join(c.data.name for c in self.sim1.player.hand),
            player_board="\t".join(c.data.name for c in self.sim1.player.characters[1:]),
            o_hp=self.sim1.opponent.hero.health,
            o_hand="\t".join(c.data.name for c in self.sim1.player.characters[1:]),
            o_board="\t".join(c.data.name for c in self.sim1.opponent.characters[1:]),
        )
        info = {
            'possible_actions': possible_actions,
            'stats': stats + "\n" + "*" * 100 + "\n" * 3
        }
        return game_observation, reward, terminal, info

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def HSenv_test():
    env = HSEnv()
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
            # print("step", step, s, r, done, random_act)


class HSsimulation(object):
    _DECK_SIZE = 30
    _MAX_CARDS_IN_HAND = 10
    _MAX_CARDS_IN_BOARD = 7

    def __init__(self):

        deck1, deck2 = self.generate_decks()
        self.player1 = Player("Agent", deck1, CardClass.MAGE.default_hero)
        self.player1.max_hand_size = self._MAX_CARDS_IN_HAND

        self.player2 = Player("Opponent", deck2, CardClass.WARRIOR.default_hero)
        self.player2.max_hand_size = self._MAX_CARDS_IN_HAND

        while True:
            try:
                new_game = Game(players=(self.player1, self.player2))
                new_game.MAX_MINIONS_ON_FIELD = self._MAX_CARDS_IN_BOARD
                # TODO: remove first_player
                new_game.start(first_player=self.player1)

                self.player = new_game.players[0]
                self.opponent = new_game.players[1]

                for player in new_game.players:
                    # TODO: offer mulligan as an action
                    cards_to_mulligan = self.mulligan_heuristic(player)
                    player.choice.choose(*cards_to_mulligan)

            except IndexError as e:
                print("init failed", e)
            else:
                self.game = new_game
                break
            self.game.logger.propagate = False

    @staticmethod
    def mulligan_heuristic(player):
        return [c for c in player.choice.cards if c.cost > 3]

    @disk_cache
    def generate_decks(self, player1_class=CardClass.MAGE, player2_class=CardClass.WARRIOR):
        while True:
            draft1 = fireplace.utils.random_draft(player1_class)
            deck1 = list(card for card in draft1 if not fireplace.cards.db[card].discover)

            draft2 = fireplace.utils.random_draft(player2_class)
            deck2 = list(card for card in draft2 if not fireplace.cards.db[card].discover)

            if len(deck1) == self._DECK_SIZE and len(deck2) == self._DECK_SIZE:
                break
        return deck1, deck2

    def terminal(self):
        return self.game.ended

    def reward(self):
        if self.player.hero.health != 0 and self.opponent.hero.health == 0:
            return 100 * (10 + 30 + (15 + 15 + 15) * 7)
        elif self.player.hero.health == 0 and self.opponent.hero.health != 0:
            return - (100 * (10 + 30 + (15 + 15 + 15) * 7))
        elif self.player.hero.health == 0 and self.opponent.hero.health == 0:
            return 0

        reward = len(self.player.hand) / self.player.max_hand_size
        reward -= len(self.opponent.hand) / self.opponent.max_hand_size

        reward += self.player.hero.health / self.player.hero._max_health
        reward -= self.opponent.hero.health / self.opponent.hero._max_health

        for entity in self.game.entities:
            if isinstance(entity, Minion):
                if entity.controller == self.player:
                    sign = +1
                else:
                    sign = -1
                reward += sign * entity.atk
                reward += sign * entity.health
            elif isinstance(entity, Secret) and entity.zone == Zone.SECRET:
                if entity.controller == self.player:
                    sign = +1
                else:
                    sign = -1
                reward += sign * entity.cost * 2
        return reward

    def actions(self):
        actions = []
        # no_target = None
        no_action = Action(None, lambda: None, {})

        for card in self.player.hand:
            if not card.is_playable():
                continue

            if card.must_choose_one:
                for choice in card.choose_cards:
                    actions.append(Action(card, card.play, {'target': target}))

            elif card.requires_target():
                for target in card.targets:
                    actions.append(Action(card, card.play, {'target': target}))
            else:
                actions.append(Action(card, card.play, {'target': None}))

        for character in self.player.characters:
            if character.can_attack():
                for enemy_char in character.targets:
                    actions.append(Action(character, character.attack, {'target': enemy_char}))
        actions += [no_action]
        return actions

    def all_actions(self):
        actions = []

        for card in self.player.hand:
            if card.requires_target():
                for target in card.targets:
                    actions.append(lambda: card.play(target=target))
            else:
                actions.append(lambda: card.play(target=None))

        for character in self.player.characters:
            for enemy_char in character.targets:
                actions.append(lambda: character.attack(enemy_char))
        return actions

    def card_to_vector(self, card):
        try:
            SPELL_TYPE = 5
            WEAPON_TYPE = 7
            if card.type == SPELL_TYPE:
                features = [card.cost, 0, 0]
            elif card.type == WEAPON_TYPE:
                features = [card.cost, card.durability, card.atk]
            else:
                features = [card.cost, card.health, card.atk]
        except Exception as e:
            import ipdb
            ipdb.set_trace()
            print("warning", e)
        return features

    def observe_player(self, player):
        if player == self.opponent:
            # print("Opponent")
            pass

        # TODO: consider all the possible permutations of the player's hand
        # FIXME: two same cards (ex CREATED_CARD) the second gets ignored same for 3..4..5. etc

        # reduce variance by sorting
        # fill with nones
        assert len(player.hand) <= self._MAX_CARDS_IN_HAND

        player_hand = list(sorted(player.hand, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_HAND
        player_hand = player_hand[:self._MAX_CARDS_IN_HAND]

        # skipping self TODO: remove?
        player_board = player.characters[1:]

        assert len(player.hand) <= self._MAX_CARDS_IN_HAND

        player_board = list(sorted(player_board, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_BOARD
        player_board = player_board[:self._MAX_CARDS_IN_BOARD]

        player_mana = player.max_mana

        game_state = player_hand + player_board + [player_mana]
        return game_state

    def observe(self):
        observation = self.observe_player(self.player)
        observation.extend(self.observe_player(self.opponent))
        return observation

    def step(self, action):
        action.use()
        observation = self.observe()
        reward = self.reward()
        terminal = self.terminal()
        return observation, reward, terminal

    def action_to_action_id(self, action):
        if isinstance(action, Action):
            action = action.vector
        return self.possible_actions[tuple(action)]

    def state_to_state_id(self, state):
        player_state, opponent_state = state
        # TODO: consider opponent state!
        # state = len(self.possible_states) * player_state + opponent_state
        state = player_state
        return self.possible_states[state]

    def sudden_death(self):
        self.player.playstate = PlayState.LOSING
        self.game.check_for_end_game()


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

    def join_game(self, simulation: HSsimulation):
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
