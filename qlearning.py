#!/usr/bin/env python3.5
import pickle
import fireplace
import random
from hearthstone.enums import CardClass, CardType
import numpy as np

from fireplace.game import Game
from fireplace.player import Player
from itertools import combinations

player_lookup = [
    'GAME_005',  # the coin
    'CREATED_CARD',
]
opponent_lookup = [
    'GAME_005',  # the coin
    'CREATED_CARD',
]
try:
    with open("lookup_tables.pkl", "rb") as fin:
        possible_hands, possible_boards = pickle.load(fin)
except IOError:
    possible_hands = list(hand for radius in range(5) for hand in combinations(range(32), radius))
    possible_boards = list(hand for radius in range(4) for hand in combinations(range(32), radius))
    with open("lookup_tables.pkl", "wb") as fout:
        pickle.dump((possible_hands, possible_boards), fout)


# http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n
def prim(n):
    sieve = [True] * int(n / 2)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = [False] * int((n - i * i - 1) / (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


def card_to_id(card, is_player):
    card_lookup = player_lookup if is_player else opponent_lookup
    try:
        idx = card_lookup.index(card.id)
    except ValueError:
        idx = card_lookup.index('CREATED_CARD')
    # return sum(ord(c) << (idx*8) for idx, c in enumerate(card.id))
    # return idx * primes[idx]
    return idx


def setup_game() -> ".game.Game":
    deck1 = fireplace.utils.random_draft(CardClass.MAGE)
    deck2 = fireplace.utils.random_draft(CardClass.WARRIOR)
    player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
    player1.max_hand_size = 4
    player2 = Player("Player2", deck2, CardClass.WARRIOR.default_hero)
    player2.max_hand_size = 4

    game = Game(players=(player1, player2))
    game.MAX_MINIONS_ON_FIELD = 3
    game.start()

    return game


class Action(object):
    def __init__(self, card, usage, params):
        self.card = card
        self.usage = usage
        self.params = params
        self.vector = [card_to_id(card, is_player=True), params['target']]

    def use(self):
        self.usage(**self.params)

    def __repr__(self):
        return "card: {}, usage: {}, params: {}".format(self.card, self.usage, self.params)


class HSsimulation(object):
    def __init__(self):
        game = setup_game()

        global primes

        self.game = game
        self.player = self.game.players[0]
        self.opponent = self.game.players[1]
        for card in self.player.deck + self.player.hand:
            player_lookup.append(card.id)
        for card in self.opponent.deck + self.opponent.hand:
            opponent_lookup.append(card.id)

    def terminal(self):
        return self.game.ended

    def reward(self):
        score = len(self.player.hand)
        score -= len(self.opponent.hand)

        score += self.player.hero.health
        score -= self.opponent.hero.health

        for entity in self.game.board:
            if entity.controller == self.player:
                sign = +1
            else:
                sign = -1
            score += sign * entity.atk
            score += sign * entity.health
        return score

    def actions(self):
        actions = []

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
                    actions.append(Action(card, card.play, {'target': None}))
                    actions.append(Action(card, character.attack, {'target': enemy_char}))
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
            import ipdb;
            ipdb.set_trace()
            print("warning", e)
        return features

    def observe_player(self, player):
        state = []
        # assert len(player.hand) < 5  # limit game to 4 card
        player_hand = tuple(sorted(set([card_to_id(card, self.player == player) for card in player.hand])))
        hand_id = possible_hands.index(player_hand)
        print("hand id:", hand_id)
        # hand_id = sum([2**idx for idx in player_hand])
        # for card in player.hand:
        #     state.extend(self.card_to_vector(card))
        player_board = tuple(sorted(set([card_to_id(card, self.player == player) for card in player.characters[1:]]))) # skipping self
        board_id = possible_boards.index(player_board)
        print("board id:", board_id)
        # for character in player.characters: #10
        #     state.extend(self.card_to_vector(card))
        # state.append(player.mana)

        # padding
        player_state = [hand_id, board_id, player.mana]
        return player_state

    def observe(self):
        state = self.observe_player(self.player)
        state.extend(self.observe_player(self.opponent))
        return state  # np.array(state)

    def step(self, action):
        observation = self.observe()
        action.use()
        next_observation = self.observe()
        reward = self.reward()
        terminal = self.terminal()
        return observation, action, next_observation, reward, terminal


def main():
    fireplace.cards.db.initialize()
    sim1 = HSsimulation()

    for player in sim1.game.players:
        print("Can mulligan %r" % player.choice.cards)
        mull_count = random.randint(0, len(player.choice.cards))
        cards_to_mulligan = random.sample(player.choice.cards, mull_count)
        player.choice.choose(*cards_to_mulligan)

    for i in range(10):
        fireplace.utils.play_turn(sim1.game)
        actions = sim1.actions()
        if len(actions) > 1:
            break

    print("terminal", sim1.terminal())
    print("reward", sim1.reward())
    actions = sim1.actions()
    print("actions", actions)
    print("all_actions", sim1.all_actions())
    print("observe", sim1.observe())
    print("acts:", actions[0])
    observation, action, next_observation, reward, terminal = sim1.step(actions[0])
    print("observation+", observation)
    print("action+", action)
    print("next_observation+", next_observation)
    print("reward+", reward)
    print("terminal+", terminal)


if __name__ == "__main__":
    main()
