#!/usr/bin/env python3.5
import hearthstone
import string
from collections import defaultdict
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


class GameActions(object):
    PASS_TURN = 0


class simple_HSEnv(gym.Env):
    def __init__(self, skip_mulligan=False):
        fireplace.cards.db.initialized = True
        print("Initializing card database")
        db, xml = hearthstone.cardxml.load()
        allowed_ids = ("GAME_005", "CS2_034", "CS2_102")
        for id, card in db.items():
            if card.description == "" or card.id in allowed_ids:
                fireplace.cards.db[id] = fireplace.cards.db.merge(id, card)

        import logging
        logging.getLogger("fireplace").setLevel(logging.ERROR)

        # HSsimulation._MAX_CARDS_IN_HAND = 2
        HSsimulation._MAX_CARDS_IN_BOARD = 2
        self.simulation = HSsimulation(skip_mulligan=skip_mulligan)

        for player in (self.simulation.player1, self.simulation.player2):
            last_card = player.hand[-1]
            if last_card.id == "GAME_005":  # "The Coin"
                last_card.discard()

        self.games_played = 0
        self.games_finished = 0

    def reset(self):
        self.actor_hero = self.simulation.player.hero
        self.games_played += 1
        possible_actions = self.simulation.actions()
        game_observation = self.simulation.observe()
        reward = self.simulation.reward()

        return game_observation, reward, False, {
            'possible_actions': possible_actions
        }

    def play_opponent_turn(self):
        fireplace.utils.play_turn(self.simulation.game)

    def step(self, action):
        terminal = False

        if action.card is None:
            try:
                self.simulation.game.end_turn()
                self.play_opponent_turn()
            except GameOver as e:
                terminal = True
        else:
            try:
                observation, reward, terminal = self.simulation.step(action)
            except GameOver as e:
                terminal = True

        possible_actions = self.simulation.actions()
        game_observation = self.simulation.observe()
        reward = self.simulation.reward()
        stats = ""
        stats += "-" * 100
        stats += "\nYOU:{player_hp}\n[{player_hand}]\n{player_board}\n{separator}\nboard:{o_board}\nENEMY:{o_hp}\n[{o_hand}]".format(
            player_hp=self.simulation.player.hero.health,
            player_hand="\t".join(c.data.name for c in self.simulation.player.hand),
            player_board="\t \t".join(c.data.name for c in self.simulation.player.characters[1:]),
            separator="_" * 100,
            o_hp=self.simulation.opponent.hero.health,
            o_hand="\t".join(c.data.name for c in self.simulation.player.characters[1:]),
            o_board="\t \t".join(c.data.name for c in self.simulation.opponent.characters[1:]),
        )
        info = {
            'possible_actions': possible_actions,
            'stats': stats + "\n" + "*" * 100 + "\n" * 3
        }
        return game_observation, reward, terminal, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class fake_HSEnv(gym.Env):
    def __init__(self, skip_mulligan=False):
        self.winner = None
        self.active_player = None

    def reset(self):
        self.winner = None
        self.active_player = random.randint(0, 1)
        info = {
            'possible_actions': self.generate_random_actions(random.randint(1, 3))
        }
        return np.random.rand(10), info

    def step(self, action):
        assert self.active_player is not None

        if np.all(action[0] == GameActions.PASS_TURN):
            self.active_player = 1 - self.active_player

        info = {
            'possible_actions': self.generate_random_actions(random.randint(1, 3))
        }
        state = np.random.rand(10)

        if random.random() < 0.01:
            terminal = True
            self.winner = random.randint(0, 1)
        else:
            terminal = False
        return state, 0, terminal, info

    def generate_random_actions(self, nr_of_actions):
        random_actions = []
        for action_idx in range(nr_of_actions):
            random_actions.append(action_idx * np.random.rand(3))
        return random_actions


def HSenv_test():
    env = simple_HSEnv(skip_mulligan=True)
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
    _card_dict = {
        'atk': None,
        # character
        'health': None,

    }
    _skipped = {
        'damage': None,
        'cost': None,
        'max_health': None,
        # Hero power
        'immune': None,
        'stealth': None,
        'secret': None,
        'overload': None,

        # spell
        'immune_to_spellpower': None,
        'receives_double_spelldamage_bonus': None,

        # enchantment
        'incoming_damage_multiplier': None,

        # weapon
        "durability": None,
        'additional_activations': (None,),
        'always_wins_brawls': (None, False),

        'has_battlecry': None,
        'cant_be_targeted_by_opponents': None,
        'cant_be_targeted_by_abilities': None,
        'cant_be_targeted_by_hero_powers': None,
        'frozen': None,
        'num_attacks': None,
        'race': None,
        'cant_attack': None,
        'taunt': None,
        'cannot_attack_heroes': None,
        # base something
        'heropower_damage': None,
        # hero
        'armor': None,
        'power': None,

        # minion
        'charge': None,
        'has_inspire': None,
        'spellpower': None,
        'stealthed': None,
        # 'always_wins_brawls': None,
        'aura': None,
        'divine_shield': None,
        'enrage': None,
        'forgetful': None,
        'has_deathrattle': None,
        'poisonous': None,
        'windfury': None,
        'silenced': None,
    }

    def __init__(self, skip_mulligan=False):

        deck1, deck2 = self.generate_decks(self._DECK_SIZE)
        self.player1 = Player("Agent", deck1, CardClass.MAGE.default_hero)
        self.player1.max_hand_size = self._MAX_CARDS_IN_HAND

        self.player2 = Player("Opponent", deck2, CardClass.WARRIOR.default_hero)
        self.player2.max_hand_size = self._MAX_CARDS_IN_HAND

        while True:
            try:
                new_game = Game(players=(self.player1, self.player2))
                new_game.MAX_MINIONS_ON_FIELD = self._MAX_CARDS_IN_BOARD
                new_game.start()

                self.player = new_game.players[0]
                self.opponent = new_game.players[1]

                if skip_mulligan:
                    cards_to_mulligan = self.mulligan_heuristic(self.player1)
                    self.player1.choice.choose(*cards_to_mulligan)

                cards_to_mulligan = self.mulligan_heuristic(self.player2)
                self.player2.choice.choose(*cards_to_mulligan)

            except IndexError as e:
                print("init failed", e)
            else:
                self.game = new_game
                break

    @staticmethod
    def mulligan_heuristic(player):
        return [c for c in player.choice.cards if c.cost > 3]

    @staticmethod
    def generate_decks(deck_size, player1_class=CardClass.MAGE, player2_class=CardClass.WARRIOR):
        while True:
            deck1 = random_draft(player1_class, max_mana=5)
            deck2 = random_draft(player2_class, max_mana=5)
            if len(deck1) == deck_size and len(deck2) == deck_size:
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

        return reward

    def actions(self):
        actions = []
        # no_target = None
        if self.player.choice:
            for card in self.player.choice.cards:
                # if not card.is_playable():
                #     continue
                if card.requires_target():
                    for target in card.targets:
                        actions.append(self.Action(card, self.player.choice.choose, {
                            # 'target': target,
                            'card': card,
                        }, self))
                else:
                    actions.append(self.Action(card, self.player.choice.choose, {
                        # 'target': None,
                        'card': card
                    }, self))
        else:
            no_action = self.Action(None, lambda: None, {}, self)

            for card in self.player.hand:
                if not card.is_playable():
                    continue

                # if card.must_choose_one:
                #     for choice in card.choose_cards:
                #         raise NotImplemented()

                elif card.requires_target():
                    for target in card.targets:
                        actions.append(self.Action(card, card.play, {'target': target}, self))
                else:
                    actions.append(self.Action(card, card.play, {'target': None}, self))

            for character in self.player.characters:
                if character.can_attack():
                    for enemy_char in character.targets:
                        actions.append(self.Action(character, character.attack, {'target': enemy_char}, self))
            actions += [no_action]
            # for action in actions:
            #     action.vector = (self.card_to_bow(action.card), self.card_to_bow(action.params))
        assert len(actions) > 0
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
        # if player == self.opponent:
        #     # print("Opponent")
        #     pass

        # TODO: consider all the possible permutations of the player's hand
        # FIXME: two same cards (ex CREATED_CARD) the second gets ignored same for 3..4..5. etc

        # reduce variance by sorting
        # fill with nones
        assert len(player.hand) <= self._MAX_CARDS_IN_HAND

        # player_hand = list(sorted(player.hand, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_HAND
        # ents = [self.entity_to_vec(c) for c in player_hand[:self._MAX_CARDS_IN_HAND]]
        # player_hand = np.hstack(ents)

        # the player hero itself is not in the board
        player_board = player.characters[1:]

        assert len(player.hand) <= self._MAX_CARDS_IN_HAND

        player_board = list(sorted(player_board, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_BOARD
        assert len(player_board) < self._MAX_CARDS_IN_BOARD or not any(player_board[self._MAX_CARDS_IN_BOARD:])
        player_board = np.hstack(self.entity_to_vec(c) for c in player_board[:self._MAX_CARDS_IN_BOARD])

        player_hero = self.entity_to_vec(player.characters[0])
        player_mana = player.max_mana

        # game_state = np.hstack((player_hand, player_board, player_hero, player_mana))
        game_state = np.hstack((player_board, player_hero, player_mana))
        return game_state

    def observe(self):
        observation = self.observe_player(self.player)
        observation = np.hstack((observation, self.observe_player(self.opponent)))
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

    @staticmethod
    def encode_to_numerical(k, val):
        if k == "power" and val:
            val = val.data.id
        if val is None:
            val = -1
        elif val is False:
            val = 0
        elif val is True:
            val = 0
        elif isinstance(val, str):
            val = HSsimulation.str_to_vec(val)

        elif isinstance(val, list):
            if len(val) != 0:
                raise Exception("wtf is this list?", val)
            val = 0
        elif isinstance(val, int):
            pass
        else:
            raise Exception("wtf is this data?", val)
        return val

    @staticmethod
    @disk_cache
    def str_to_vec(val):
        @disk_cache
        def load_text_map():
            fireplace.cards.db.initialize()
            descs = set()
            for card_id, card_obj in fireplace.cards.db.items():
                descs.add(card_obj.description.replace("\n", " "))
            wf = defaultdict(int)
            for d in descs:
                table = d.maketrans({key: " " for key in string.punctuation})
                d = d.translate(table).lower()
                for word in d.split(" "):
                    if word != "":
                        wf[word] += 1
            text_map = []
            reverse_text_map = {}
            for word in sorted(wf, key=lambda x: wf[x], reverse=True):
                if wf[word] > 4:
                    text_map.append(word)
                    reverse_text_map[word] = len(text_map)
            return text_map, reverse_text_map

        text_map, reverse_text_map = load_text_map()
        bag_of_words = np.zeros(len(text_map))
        table = val.maketrans({key: " " for key in string.punctuation})
        val = val.translate(table).lower().replace("\n", " ")
        for word in val.split(" "):
            try:
                bag_of_words[reverse_text_map[word]] += 1
            except KeyError:
                pass
        return bag_of_words

    def player_to_bow(self, entity):
        # TODO: check all the possible attributes
        player_dict = {
            'combo': None,
            'fatigue_counter': None,
            'healing_as_damage': None,
            'healing_double': None,
            'shadowform': None,
            'times_hero_power_used_this_game': None,
        }
        player_lst = []
        for k in sorted(player_dict.keys()):
            try:
                val = player_obj.__getattribute__(k)
            except AttributeError:
                val = None
            val = encode_to_numerical(k, val)
            player_lst.append(val)
        return player_lst

    def card_to_bow(self, card_obj):
        assert card_obj is None or card_obj.data.description == ""

        card_dict = self._card_dict  # .copy()
        # TODO: check all the possible attributes
        for k in card_dict:
            try:
                card_dict[k] = card_obj.__getattribute__(k)
            except AttributeError:
                card_dict[k] = None

        # crash if skipping important data
        for k in self._skipped:
            try:
                value = card_obj.__getattribute__(k)
            except AttributeError:
                pass
            else:
                assert self._skipped[k] is None or value in self._skipped[k]

        # card_dict['description'] = ''

        card_lst = []
        for k in sorted(card_dict.keys()):
            val = card_dict[k]
            # print(k, val)

            val = self.encode_to_numerical(k, val)
            if isinstance(val, int):
                card_lst.append(val)
            elif isinstance(val, np.ndarray):
                raise NotImplementedError("simple environment doesn't support complex cards")
                # card_lst.extend(list(val))
            else:
                raise TypeError()
        assert len(card_lst) == 2
        return np.array(card_lst)

    def entity_to_vec(self, entity):
        return self.card_to_bow(entity)


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


def random_draft(card_class: CardClass, exclude=set(), deck_length=30, max_mana=30):
    from fireplace import cards

    deck = []
    collection = []
    hero = card_class.default_hero

    for card_id, card_obj in cards.db.items():
        if card_obj.description != '':
            continue
        if card_id in exclude:
            continue
        if not card_obj.collectible:
            continue
        if card_obj.type == CardType.HERO:
            # Heroes are collectible...
            continue
        if card_obj.card_class and card_obj.card_class not in (card_class, CardClass.NEUTRAL):
            continue
        if card_obj.cost > max_mana:
            continue
        collection.append(card_obj)

    while len(deck) < deck_length:
        card = random.choice(collection)
        if deck.count(card.id) < card.max_count_in_deck:
            deck.append(card.id)

    return deck


if __name__ == "__main__":
    HSenv_test()
