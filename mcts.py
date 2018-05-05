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
from itertools import combinations, product, permutations, count


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


class HSsimulation(object):
    _DECK_SIZE = 15
    _MAX_CARDS_IN_HAND = 3
    _MAX_CARDS_IN_BOARD = 2

    def __init__(self, possible_states=None, possible_actions=None):
        self.score = 0
        while True:
            self.player_lookup = {
                'NO_TARGET': 0,
                'GAME_005': 1,  # the coin
                'CREATED_CARD': 2,
            }
            self.opponent_lookup = {
                'NO_TARGET': 0,
                'GAME_005': 1,  # the coin OR or ENEMY FACE
                'CREATED_CARD': 2,
            }
            self.EXTRA_IDS = len(self.player_lookup)
            try:
                self.game = self.setup_game()
                self.player = self.game.players[0]
                self.opponent = self.game.players[1]
                self.skip_the_boring_parts()
            except IndexError:
                # print("coin failed")
                pass
            else:
                break
            self.game.logger.propagate = False
        # self.possible_hands, self.possible_boards = self.initialize_lookup_tables()
        if not possible_actions or not possible_states:
            print("SLOOOW INIT")
            self.possible_states, self.possible_actions = self.initialize_lookup_tables()
        else:
            self.possible_states, self.possible_actions = possible_states, possible_actions

    def initialize_lookup_tables(self):
        try:
            # raise IOError
            with open("lookup_tables.pkl", "rb") as fin:
                possible_states, possible_actions = pickle.load(fin)
        except IOError:
            possible_states = {}
            cards_in_deck = set(range(1, self.EXTRA_IDS + self._DECK_SIZE))

            counter = 0
            for hand in combinations(list(cards_in_deck) + [0] * self._MAX_CARDS_IN_HAND, self._MAX_CARDS_IN_HAND):
                cards_left_in_deck = cards_in_deck.difference(hand)
                # TODO: remove spells from the possible board state; should reduce from 15 choose 4 to ~7 choose 4!
                for board in combinations(list(cards_left_in_deck) + [0] * self._MAX_CARDS_IN_BOARD,
                                          self._MAX_CARDS_IN_BOARD):
                    for mana in range(1, 11):
                        state = sorted(hand) + sorted(board) + [mana]
                        counter += 1
                        possible_states[tuple(state)] = counter
            possible_actions = {}
            # (len(simulation.player_lookup) + 1) * (len(simulation.opponent_lookup) + 1)  # 1s are for players' hero
            cards_in_deck = set(range(len(self.player_lookup)))
            counter = 0
            for action_source in cards_in_deck:
                # HACK: ATTACKING NO TARGET MEANS ENEMY FACE if no-target is not an option
                # FIXME: BUG: CANNOT TARGET OWN MINIONS (like with buffs)
                for action_target in range(len(self.opponent_lookup)):
                    counter += 1
                    possible_actions[(action_source, action_target)] = counter

            with open("lookup_tables.pkl", "wb") as fout:
                print(len(possible_states), "states", len(possible_actions), "actions")
                pickle.dump((possible_states, possible_actions), fout)
        return possible_states, possible_actions

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

        offset = max(self.player_lookup.values())
        new_cards = {(str(card), offset + idx + 1) for idx, card in enumerate(deck1)}
        self.player_lookup.update(new_cards)
        offset = max(self.opponent_lookup.values())
        new_cards = {(str(card), offset + idx + 1) for idx, card in enumerate(deck2)}
        self.opponent_lookup.update(new_cards)

        player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
        player1.max_hand_size = self._MAX_CARDS_IN_HAND
        player2 = Player("Player2", deck2, CardClass.WARRIOR.default_hero)
        player2.max_hand_size = self._MAX_CARDS_IN_HAND

        new_game = Game(players=(player1, player2))
        new_game.MAX_MINIONS_ON_FIELD = self._MAX_CARDS_IN_BOARD
        new_game.start(first_player=player1)
        return new_game

    def card_to_id(self, card, is_player):
        card_lookup = self.player_lookup if is_player else self.opponent_lookup
        try:
            idx = card_lookup[card.id]
        except KeyError:
            idx = card_lookup['CREATED_CARD']
        # return sum(ord(c) << (idx*8) for idx, c in enumerate(card.id))
        # return idx * primes[idx]
        return idx

    def terminal(self):
        return self.game.ended

    def reward(self):
        if self.player.hero.health != 0 and self.opponent.hero.health == 0:
            return 100*(10+30+(15+15+15)*7)

        score = len(self.player.hand)
        score -= len(self.opponent.hand)

        score += self.player.hero.health
        score -= self.opponent.hero.health

        for entity in self.game.entities:
            if isinstance(entity, Minion):
                if entity.controller == self.player:
                    sign = +1
                else:
                    sign = -1
                score += sign * entity.atk
                score += sign * entity.health
            elif isinstance(entity, Secret) and entity.zone == Zone.SECRET:
                if entity.controller == self.player:
                    sign = +1
                else:
                    sign = -1
                score += sign * entity.cost * 2
        reward = score - self.score
        self.score = score
        return reward

    def actions(self):
        actions = []
        no_target = self.player_lookup['NO_TARGET']
        no_action = Action(self.player_lookup['NO_TARGET'], None, lambda: None, {}, self.player_lookup['NO_TARGET'])

        for card in self.player.hand:
            card_id = self.card_to_id(card, True)
            if card.is_playable():
                # if card.must_choose_one:
                #     card = random.choice(card.choose_cards)

                if card.requires_target():
                    for target in card.targets:
                        target_id = self.card_to_id(target, True)
                        actions.append(Action(card_id, card, card.play, {'target': target}, target_id))
                else:
                    actions.append(Action(card_id, card, card.play, {'target': None}, target=no_target))

        for character in self.player.characters:
            char_id = self.card_to_id(character, True)
            if character.can_attack():
                for enemy_char in character.targets:
                    target_id = self.card_to_id(enemy_char, True)
                    actions.append(Action(char_id, character, character.attack, {'target': enemy_char}, target_id))
        actions = [no_action] + [a for a in actions if 2 not in a.vector]
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
            # FIXME considering opponent hand empty instead of unknown
            player_hand = []
        else:
            player_hand = [self.card_to_id(card, self.player == player) for card in player.hand]
        # pad hand with nones
        empty_token = self.player_lookup['NO_TARGET']
        # FIXME: two same cards (ex CREATED_CARD) the second gets ignored same for 3..4..5. etc
        player_hand = list(
            sorted((list(set(player_hand)) + [empty_token] * self._MAX_CARDS_IN_HAND)[:self._MAX_CARDS_IN_HAND]))
        # skipping self
        player_board = [self.card_to_id(card, self.player == player) for card in player.characters[1:]]
        # FIXME: two same cards (ex CREATED_CARD) the second gets ignored same for 3..4..5. etc
        player_board = list(
            sorted(list(set(player_board)) + [empty_token] * self._MAX_CARDS_IN_BOARD)[:self._MAX_CARDS_IN_BOARD])

        player_mana = player.max_mana
        if player == self.opponent:
            player_mana += 1
        game_state = player_hand + player_board + [player_mana]
        board_id = self.possible_states[tuple(game_state)]
        return board_id

    def observe(self):
        player_state = self.observe_player(self.player)
        # TODO: consider opponent state
        # opponent_state = self.observe_player(self.opponent)
        # return [player_state, opponent_state]  # np.array(state)
        return player_state

    def step(self, action):
        observation = self.observe()
        action.use()
        next_observation = self.observe()
        reward = self.reward()
        terminal = self.terminal()
        return observation, action, next_observation, reward, terminal

    def skip_the_boring_parts(self):
        for player in self.game.players:
            # print("Can mulligan %r" % player.choice.cards)
            if player == self.player:
                cards_to_mulligan = [c for c in player.choice.cards if c.cost > 3]
            else:
                mull_count = random.randint(0, len(player.choice.cards))
                cards_to_mulligan = random.sample(player.choice.cards, mull_count)
            player.choice.choose(*cards_to_mulligan)
            # for i in range(10):
            #     fireplace.utils.play_turn(self.game)
            #     actions = self.actions()
            #     if len(actions) > 1:
            #         break

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
        self.learning_rate = 0.99
        self.gamma = 0.99

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
                for nearby_state in range(int(state_id), int(state_id) + 100):
                    try:
                        approximate_q = self.Q_table[str(nearby_state)][action_id]
                        return approximate_q
                    except KeyError:
                        pass
                self.Q_table[state_id][action_id] = 0.0
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


def main():
    fireplace.cards.db.initialize()
    # logger = fireplace.logging.getLogger('fireplace')
    import logging
    logging.getLogger("fireplace").setLevel(logging.ERROR)
    player = Agent()
    sim1 = HSsimulation()
    cached_lookup_tables = (sim1.possible_states, sim1.possible_actions)
    games_played = 0
    games_won = 0
    games_finished = 0
    old_miss = 0
    print("games_played, win %, result, abs_change, turn, q_hit_ratio")
    while True:
        change = 0
        abs_change = 0
        count = 0
        player_actions = 0
        player_reward = 0
        try:
            sim1 = HSsimulation(*cached_lookup_tables)
            player.join_game(sim1)
            actor_hero = sim1.game.current_player.hero
            games_played += 1

            while True:
                actions = sim1.actions()

                if sim1.game.turn > 15:
                    sim1.sudden_death()

                if len(actions) == 1:
                    # end turn
                    # sim1.game.end_turn()
                    assert actor_hero == sim1.game.current_player.hero
                    fireplace.utils.play_turn(sim1.game)
                    assert actor_hero != sim1.game.current_player.hero

                    # play opponent turn
                    fireplace.utils.play_turn(sim1.game)
                    assert actor_hero == sim1.game.current_player.hero
                else:
                    observation = sim1.observe()
                    choosen_action = player.choose_action(observation, actions)

                    observation, action, next_observation, reward, terminal = sim1.step(choosen_action)
                    player_reward += reward
                    player_actions += 1

                    # print(choosen_action)
                    # print(reward)

                    delta = player.learn_from_reaction(observation, action, reward, next_observation)
                    change += delta
                    abs_change += abs(delta)
                    count += 1

                    if terminal:
                        break
        except GameOver as e:
            print_row = False
            if sim1.game.turn != 17:
                games_won += 1 if sim1.player.playstate == PlayState.WON else 0
                games_finished += 1
                print_row = True
            elif games_finished == 0:
                games_finished = 1

            row = "\t".join(map(str, (
                0.0000000000000001 + games_won / games_finished,
                sim1.player.playstate,
                int(abs_change),
                sim1.game.turn,
                player.qhit / (player.qmiss + player.qhit),
                player.qmiss - old_miss,
                0 if player_actions == 0 else player_reward/player_actions,
                player_actions,
            )))
            with open("metrics.tsv", "a") as fout:
                fout.write(row + "\n")
            """
            print(games_played,
                  0.0000000000000001 + games_won / games_finished,
                  sim1.player.playstate,
                  int(abs_change),
                  sim1.game.turn,
                  player.qhit / (player.qmiss + player.qhit),
                  player.qmiss - old_miss,
                  sep="\t")
            """
            if print_row: print(row)
            old_miss = player.qmiss

            # print("observation+", observation, "action+", action, "next_observation+", next_observation, "reward+",
            #  reward)
            # print("terminal+", terminal)


if __name__ == "__main__":
    main()


class MCTS(object):
    def __init__(self, environment, player_neural_network):
        return None

    def reset(self):
        pass