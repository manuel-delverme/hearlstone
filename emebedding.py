#!/usr/bin/env python3.5
from utils import disk_cache
import bz2
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

try:
    with open("data/string_encoding.pkl", "rb") as fin:
        encoded_strings, str_encoding = pickle.load(fin)
except FileNotFoundError:
    encoded_strings, str_encoding = {}, []


class Action(object):
    def __init__(self, card_object, usage, params):
        self.card_obj = card_object
        self.usage = usage
        self.params = params
        if card_object:
            target_card = None
            if params['target']:
                target_card = params['target']
            self.bow = [card_object.id, card_to_bow(target_card)]

    def use(self):
        self.usage(**self.params)

    def __repr__(self):
        return "card: {}, usage: {}".format(self.card_obj, self.params)


def player_to_bow(player_obj):
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


# @disk_cache
def card_to_bow(card_obj, exact=False):
    if not exact:
        return card_to_bow_lossy(card_obj)
    # TODO: check all the possible attributes
    card_dict = {
        'atk': None,
        # 'entourage': None,
        'has_battlecry': None,

        # character
        'health': None,
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
        'always_wins_brawls': None,
        'aura': None,
        'divine_shield': None,
        'enrage': None,
        'forgetful': None,
        'has_deathrattle': None,
        'poisonous': None,
        'windfury': None,
        'silenced': None,

        'cost': None,
        'damage': None,
        'immune': None,
        'max_health': None,
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

        # Hero power
        'additional_activations': None,
    }
    for k in card_dict:
        try:
            card_dict[k] = card_obj.__getattribute__(k)
        except AttributeError:
            card_dict[k] = None
    try:
        card_dict['description'] = card_obj.data.description
    except AttributeError:
        card_dict['description'] = None

    card_lst = []
    for k in sorted(card_dict.keys()):
        val = card_dict[k]
        val = encode_to_numerical(k, val)

        card_lst.append(val)

    return np.array(card_lst)


def card_to_bow_lossy(card_obj):
    card_lst = [-1, -1, -1]
    if card_obj is None:
        return card_lst
    elif card_obj == "UNK":
        return [-1, -1, "UNK"]
    try:
        card_lst[0] = card_obj.atk
    except AttributeError:
        card_lst[0] = -1

    try:
        card_lst[1] = card_obj.health
    except AttributeError:
        card_lst[1] = -1

    card_lst[2] = card_obj.id
    return np.array(card_lst)


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
        try:
            val = str_encoding[val]
        except KeyError:
            encoded_strings.append(val)
            str_encoding[val] = len(encoded_strings)
            with open("data/string_encoding.pkl", "wb") as fout:
                pickle.dump((encoded_strings, str_encoding), fout)
            val = str_encoding[val]
    elif isinstance(val, list):
        if len(val) != 0:
            raise Exception("wtf is this list?", val)
        val = 0
    elif isinstance(val, int):
        pass
    else:
        raise Exception("wtf is this data?", val)
    return val


def setup_game() -> ".game.Game":
    draft1 = fireplace.utils.random_draft(CardClass.MAGE)
    draft2 = set(fireplace.utils.random_draft(CardClass.WARRIOR))
    player1 = Player("Player1", draft1, CardClass.MAGE.default_hero)
    player2 = Player("Player2", draft2, CardClass.WARRIOR.default_hero)
    new_game = Game(players=(player1, player2))
    new_game.start()
    return new_game


class HSsimulation(object):
    def __init__(self):
        while True:
            try:
                self.game = setup_game()
                self.player = self.game.players[0]
                self.opponent = self.game.players[1]
                self.skip_the_boring_parts()
            except IndexError:
                pass
            else:
                break

    def terminal(self):
        return self.game.ended

    def actions(self):
        actions = []
        no_action = None

        for card in self.player.hand:
            if card.is_playable():
                # if card.must_choose_one:
                #     card = random.choice(card.choose_cards)
                if card.requires_target():
                    for target in card.targets:
                        if card.must_choose_one:
                            for choice in card.choose_cards:
                                if choice.requires_target():
                                    actions.append(Action(card, card.play, {'target': target, 'choice': choice}))
                                else:
                                    actions.append(Action(card, card.play, {'target': None, 'choice': choice}))
                        else:
                            actions.append(Action(card, card.play, {'target': target}))
                else:
                    target = None
                    if card.must_choose_one:
                        for choice in card.choose_cards:
                            if choice.requires_target():
                                for target in card.targets:
                                    actions.append(Action(card, card.play, {'target': target, 'choice': choice}))
                            else:
                                actions.append(Action(card, card.play, {'target': target, 'choice': choice}))
                    else:
                        actions.append(Action(card, card.play, {'target': target}))

        for character in self.player.characters:
            if character.can_attack():
                for enemy_char in character.targets:
                    actions.append(Action(character, character.attack, {'target': enemy_char}))
        actions = [no_action] + actions
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

    def observe_player(self, player):
        player_state = player_to_bow(player)
        empty_token = None

        if player == self.opponent:
            player_hand = ['UNK' for _ in player.hand]
            player_hand = (player_hand + [empty_token] * player.max_hand_size)[:player.max_hand_size]
        else:
            player_hand = player.hand
            player_hand = list(sorted(player_hand, key=lambda c: c.id) +
                               [empty_token] * player.max_hand_size)[:player.max_hand_size]

        for card_in_hand in player_hand:
            card_bow = card_to_bow(card_in_hand)
            player_state.append(card_bow)

        # skipping self
        # FIXME: two same cards (ex CREATED_CARD) the second gets ignored same for 3..4..5. etc
        player_board = [card for card in player.characters]
        player_board = list(sorted(player_board, key=lambda c: c.id) +
                            [empty_token] * player.minion_slots)[:player.minion_slots + 1]  # player is part of board

        for card_in_board in player_board:
            card_bow = card_to_bow(card_in_board)
            player_state.append(card_bow)

        player_mana = player.max_mana
        if player == self.opponent:
            player_mana += 1
        player_state.append(player_mana)
        return np.hstack(player_state)

    def observe(self):
        player_state = self.observe_player(self.player)
        opponent_state = self.observe_player(self.opponent)
        return np.hstack((player_state, opponent_state))

    def step(self, action):
        if self.player.choice:
            print("SOMEONE LEFT A CHOICE OPEN; CLOSING")
            choice = random.choice(self.player.choice.cards)
            self.player.choice.choose(choice)
        observation = self.observe()
        if action:
            action.use()
            if self.player.choice:
                choice = random.choice(self.player.choice.cards)
                self.player.choice.choose(choice)
            next_observation = self.observe()
        else:
            next_observation = observation
        terminal = self.terminal()
        return observation, action, next_observation, terminal

    def skip_the_boring_parts(self):
        for player in self.game.players:
            mull_count = random.randint(0, len(player.choice.cards))
            cards_to_mulligan = random.sample(player.choice.cards, mull_count)
            player.choice.choose(*cards_to_mulligan)


def main():
    games_finished = 0
    nr_tuples = 0
    file_idx = 0
    data_dump = "training_data{}.pbz"
    while True:
        try:
            with open(data_dump.format(file_idx), "rb") as _:
                pass
        except FileNotFoundError:
            break
        else:
            file_idx += 1

    with bz2.BZ2File(data_dump.format(file_idx), "wb") as fout:
        fireplace.cards.db.initialize()
        training_set = []
        while True:
            try:
                sim1 = HSsimulation()
                while True:
                    actions = sim1.actions()
                    if len(actions) == 1:
                        # end turn
                        fireplace.utils.play_turn(sim1.game)
                        # play opponent turn
                        fireplace.utils.play_turn(sim1.game)
                    else:
                        choosen_action = random.choice(actions)
                        observation, action, next_observation, terminal = sim1.step(choosen_action)
                        if choosen_action is not None:
                            training_tuple = (observation, action.bow, next_observation)
                            training_set.append(training_tuple)
            except GameOver as e:
                games_finished += 1
                # if games_finished > 10:
                #     break
                nr_tuples += len(training_set)
                print(games_finished, nr_tuples)
                pickle.dump(training_set, fout)
                fout.flush()
                training_set = []
            except TypeError as e:
                print("game failed")
            except Exception as e:
                print(str(e))
                break


if __name__ == "__main__":
    main()

with tf.Session() as sess:
    sess.run(init)

    for i in range(num_steps):
        s0, a0 = get_data(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: s0})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

        # g = sess.run(decoder_op, feed_dict={X: batch_x})
