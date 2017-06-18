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


class Action(object):
    def __init__(self, card_object, usage, params):
        self.card_obj = card_object
        self.usage = usage
        self.params = params
        if card_object:
            target_id = None
            if params['target']:
                target_id = params['target'].id
            self.bow = [card_object.id, target_id]

    def use(self):
        self.usage(**self.params)

    def __repr__(self):
        return "card: {}, usage: {}".format(self.card_obj, self.params)


def card_to_bow(card_obj):
    # TODO: check all the possible attributes
    card_dict = {
        'atk': None,
        'entourage': None,
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

    return card_dict


class HSsimulation(object):
    def __init__(self):
        while True:
            try:
                self.game = self.setup_game()
                self.player = self.game.players[0]
                self.opponent = self.game.players[1]
                self.skip_the_boring_parts()
            except IndexError:
                pass
            else:
                break

    def setup_game(self) -> ".game.Game":
        try:
            with open("decks", "rb") as fin:
                deck1, deck2 = pickle.load(fin)
        except IOError:
            draft1 = fireplace.utils.random_draft(CardClass.MAGE)
            deck1 = list(card for card in set(draft1) if not fireplace.cards.db[card].discover)[:15]
            draft2 = set(fireplace.utils.random_draft(CardClass.WARRIOR))
            deck2 = list(card for card in draft2 if not fireplace.cards.db[card].discover)[:15]
            assert len(deck1) == 15
            assert len(deck2) == 15
            with open("decks", "wb") as fout:
                pickle.dump((deck1, deck2), fout)

        player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
        player2 = Player("Player2", deck2, CardClass.WARRIOR.default_hero)
        new_game = Game(players=(player1, player2))
        new_game.start()
        return new_game

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
                        actions.append(Action(card, card.play, {'target': target}))
                else:
                    actions.append(Action(card, card.play, {'target': None}))

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
        player_state = {}
        if player == self.opponent:
            # FIXME considering opponent hand empty instead of unknown
            player_hand = []
        else:
            player_hand = player.hand

        # pad hand with nones
        empty_token = None

        player_hand = list(sorted(player_hand, key=lambda c: c.id) +
                           [empty_token] * player.max_hand_size)[:player.max_hand_size]
        player_state['hand'] = []
        for card_in_hand in player_hand:
            card_bow = card_to_bow(card_in_hand)
            player_state['hand'].append(card_bow)

        # skipping self
        # FIXME: two same cards (ex CREATED_CARD) the second gets ignored same for 3..4..5. etc
        player_board = [card for card in player.characters]
        player_board = list(sorted(player_board, key=lambda c: c.id) +
                            [empty_token] * player.minion_slots)[:player.minion_slots]

        player_state['board'] = []
        for card_in_board in player_board:
            card_bow = card_to_bow(card_in_board)
            player_state['board'].append(card_bow)

        player_mana = player.max_mana
        if player == self.opponent:
            player_mana += 1
        player_state['mana'] = player_mana
        return player_state

    def observe(self):
        player_state = self.observe_player(self.player)
        # TODO: consider opponent state
        # opponent_state = self.observe_player(self.opponent)
        # return [player_state, opponent_state]  # np.array(state)
        return player_state

    def step(self, action):
        observation = self.observe()
        if action:
            action.use()
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
                    if choosen_action.card_obj is not None:
                        training_tuple = (observation, action.bow, next_observation)
                        training_set.append(training_tuple)
        except GameOver as e:
            pass


if __name__ == "__main__":
    main()
