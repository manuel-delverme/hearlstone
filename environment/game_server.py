import zerorpc
import logging
import utils
import string
from hearthstone.enums import CardClass, CardType
from utils import disk_cache
import fireplace.cards
from fireplace.player import Player
from fireplace.game import Game, PlayState, Zone

class IllegalMove(Exception):
    pass

class HSsimulation(object):
    _DECK_SIZE = 30
    _MAX_CARDS_IN_HAND = 6
    _MAX_CARDS_IN_BOARD = 3
    _card_dict = {
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
    }
    _skipped = {
        # Hero power
        'additional_activations': (None,),
        'always_wins_brawls': (None, False),
    }

    def __init__(self):
        self.game = None
        self.action_table = {}

        fireplace.cards.db.initialize()
        self.reset()

    @classmethod
    def get_nr_possible_actions(cls):
        # 2 is the player's char and the heropower
        nr_sources = cls._MAX_CARDS_IN_HAND + cls._MAX_CARDS_IN_BOARD + 2

        # 2 is the opponent's char and NO_TARGET
        nr_targets = cls._MAX_CARDS_IN_BOARD + 2
        return nr_sources * nr_targets


    @staticmethod
    def mulligan_heuristic(player):
        return [c for c in player.choice.cards if c.cost > 3]

    @staticmethod
    @disk_cache
    def generate_decks(deck_size, player1_class=CardClass.MAGE, player2_class=CardClass.WARRIOR):
        while True:
            draft1 = fireplace.utils.random_draft(player1_class)
            deck1 = list(card for card in draft1 if not fireplace.cards.db[card].discover)

            draft2 = fireplace.utils.random_draft(player2_class)
            deck2 = list(card for card in draft2 if not fireplace.cards.db[card].discover)

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

        # for entity in self.game.entities:
        #     if isinstance(entity, Minion):
        #         if entity.controller == self.player:
        #             sign = +1
        #         else:
        #             sign = -1
        #         reward += sign * entity.atk / 12
        #         reward += sign * entity.health / 12
        #     elif isinstance(entity, Secret) and entity.zone == Zone.SECRET:
        #         if entity.controller == self.player:
        #             sign = +1
        #         else:
        #             sign = -1
        #         reward += sign * entity.cost * 2
        return reward

    def actions(self):
        """
            returns the available actions for this game state
            and caches the values for decoding
        """

        if self.player.choice:
            raise NotImplementedError("choice not supported")
            # for card in self.player.choice.cards: etc.

        possible_actions = {}

        possible_actions['pass'] = self.Action(None, lambda: None, {})

        # check players hand for possible actions
        for hand_pos_idx, card in enumerate(self.player.hand):
            if not card.is_playable():
                continue

            action_name = "play_hand_{}".format(hand_pos_idx)

            if card.requires_target():
                for target_idx, target in enumerate(card.targets):
                    action_id = "{}_on_board_{}".format(action_name, target_idx)
                    action_obj = self.Action(card, card.play, {'target': target})
                    possible_actions[action_id] =  action_obj
            else:
                possible_actions[action_name] = self.Action(card, card.play, {'target': None})

        # check hero power
        hero_power = self.player.hero.power
        if hero_power.is_playable() and hero_power.is_usable():
            action_name = 'hero_power'
            if not hero_power.requires_target():
                possible_actions[action_name] = self.Action(hero_power, hero_power.use, {})
            else:
                for target_idx, target in enumerate(hero_power.targets):
                    action_id = "{}_on_board_{}".format(action_name, target_idx)
                    action_obj = self.Action(hero_power, hero_power.use,
                            {'target': target})

                    possible_actions[action_id] =  action_obj

        # weapon attack
        if self.player.hero.can_attack():
            action_name = "hero_attacks"
            for enemy_char in self.player.hero.targets:
                action_id = "{}_board_{}".format(action_name, target_idx)
                possible_actions[action_id] = self.Action(self.player.hero, self.player.hero.attack, {'target': enemy_char})

        self.action_table.clear()
        for action_id, action in possible_actions.items():
            self.action_table[action_id] = action
        print(possible_actions)
        return tuple(possible_actions.keys())

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
        def __init__(self, card, usage, params):
            self.card = card
            self.usage = usage
            self.params = params

        def use(self):
            self.usage(**self.params)

        def __repr__(self):
            return "card: {}, usage: {}, vector: {}".format(self.card, self.params, None)

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

        player_hand = list(sorted(player.hand, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_HAND
        flatten_player_hand = []
        for c in player_hand[:self._MAX_CARDS_IN_HAND]:
            flatten_player_hand.extend(self.entity_to_vec(c))
        player_hand = tuple(flatten_player_hand)

        # the player hero itself is not in the board
        player_board = player.characters[1:]

        assert len(player.hand) <= self._MAX_CARDS_IN_HAND

        player_board = list(sorted(player_board, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_BOARD
        assert len(player_board) < self._MAX_CARDS_IN_BOARD or not any(player_board[self._MAX_CARDS_IN_BOARD:])
        flatten_player_board = []
        for c in player_board[:self._MAX_CARDS_IN_BOARD]:
            flatten_player_board.extend(self.entity_to_vec(c))
        player_board = tuple(flatten_player_board)

        player_entity = player.characters[0]
        player_hero = self.entity_to_vec(player_entity)
        player_mana = player.max_mana

        # print(player_hand.shape, player_board.shape, player_hero.shape)
        game_state = player_hand + player_board + player_hero + (player_mana,)
        return game_state

    def observe(self):
        observation = self.observe_player(self.player)
        observation = observation + self.observe_player(self.opponent)
        return observation

    def step(self, action):
        if self.game.ended:
            raise IllegalMove

        # action = utils.to_tuples(action)
        action_obj = self.action_table[action]
        print("[USING]", action_obj)
        action_obj.use()
        return self.observe_gamestate()

    def end_turn(self):
        self.game.end_turn()

    def play_random_turn(self):
        fireplace.utils.play_turn(self.game)

    def reset(self):
        if self.game is not None:
            del self.game

        deck1, deck2 = self.generate_decks(self._DECK_SIZE)
        self.player1 = Player("Agent", deck1, CardClass.MAGE.default_hero)
        self.player1.max_hand_size = self._MAX_CARDS_IN_HAND

        self.player2 = Player("Opponent", deck2, CardClass.WARRIOR.default_hero)
        self.player2.max_hand_size = self._MAX_CARDS_IN_HAND

        # self.actor_hero = self.game.current_player.hero
        while True:
            try:
                new_game = Game(players=(self.player1, self.player2))
                # new_game.logger.propagate = False
                new_game.logger.setLevel(logging.WARNING)
                new_game.MAX_MINIONS_ON_FIELD = self._MAX_CARDS_IN_BOARD
                new_game.start()

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
        return self.observe_gamestate()

    def observe_gamestate(self):
        possible_actions = self.actions()
        game_observation = self.observe()
        reward = self.reward()
        return game_observation, reward, self.terminal(), {
            'possible_actions': possible_actions
        }

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
        elif type(val) == str:
            val = HSsimulation.str_to_vec(val)
        elif type(val) == list:
            if len(val) != 0:
                raise Exception("wtf is this list?", val)
            val = 0
        elif type(val) == int:
            pass
        elif isinstance(val, int):
            val = int(val)
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
        bag_of_words = [0, ] * len(text_map)
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
                assert value in self._skipped[k]

        try:
            card_dict['description'] = card_obj.data.description
        except AttributeError:
            card_dict['description'] = ''

        card_lst = []
        for k in sorted(card_dict.keys()):
            val = card_dict[k]
            val = self.encode_to_numerical(k, val)
            if isinstance(val, int):
                card_lst.append(val)
            elif isinstance(val, list):
                card_lst.extend(val)
            else:
                raise TypeError(type(val))
        if isinstance(card_obj, fireplace.card.Hero):
            # assert len(card_lst) == 634
            assert len(card_lst) == 856
        else:
            # assert len(card_lst) == 337
            assert len(card_lst) == 448
        return tuple(card_lst)

    def entity_to_vec(self, entity):
        return self.card_to_bow(entity)

    def get_player_hero(self):
        return self.game.current_player.hero

    def get_player_hero_health(self):
        return self.game.current_player.hero.health
    
    def get_ascii_board(self):
        stats = ""
        stats += "-" * 100
        stats += "\nYOU:{player_hp}\nH:{player_hand}\n[{player_board}]\n[{o_board}]\nENEMY:{o_hp}\thand:{o_hand}".format(
            player_hp=self.player.hero.health,
            player_hand="\t".join(c.data.name for c in self.player.hand),
            player_board="\t".join(c.data.name for c in self.player.characters[1:]),
            o_hp=self.opponent.hero.health,
            o_hand="\t".join(c.data.name for c in self.player.characters[1:]),
            o_board="\t".join(c.data.name for c in self.opponent.characters[1:]),
        ) + "\n" + "*" * 100 + "\n" * 3
        return stats



def main():
    obj = HSsimulation()
    s = zerorpc.Server(obj)
    print("binding 31337..")
    s.bind("tcp://0.0.0.0:31337")
    print("done")
    s.run()
