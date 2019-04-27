import functools
import random
import shelve
import string
from collections import defaultdict, OrderedDict

import fireplace.game
import numpy as np
from fireplace.game import PlayState
from fireplace.player import Player
from hearthstone.enums import CardClass, Zone

import hs_config
from shared import utils


def can_attack(self):
  if not self._zone == Zone.PLAY:
    return False
  if not self.attack_targets:
    return False
  if self.exhausted:
    return False
  if not self.atk:
    return False
  return True


class CoinRules:
  """
  Randomly determines the starting player when the Game starts.
  The second player gets "The Coin" (GAME_005).
  """

  def pick_first_player(self):
    if hs_config.VanillaHS.always_first_player:
      winner = self.players[0]
      self.log("[FORCED] Player starts", winner)
    else:
      winner = random.choice(self.players)
      self.log("Tossing the coin... %s wins!", winner)
    return winner, winner.opponent

  def begin_turn(self, player):
    # if self.turn == 0:
    # self.log("%s gets The Coin (%s)", self.player2, THE_COIN)
    # self.player2.give(THE_COIN)
    super().begin_turn(player)


class Game(fireplace.game.MulliganRules, CoinRules, fireplace.game.BaseGame):
  pass


class HSsimulation(object):
  _DECK_SIZE = 30
  _MAX_CARDS_IN_HAND = 10
  _MAX_CARDS_IN_BOARD = 7
  _card_dict = OrderedDict((
    ('atk', None),
    # character
    ('health', None),
    ('can_attack', None),
  ))
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

  def __init__(self, skip_mulligan=False, cheating_opponent=False, sort_decks=True, starting_hp=None):
    player1_class = CardClass.MAGE.default_hero
    player2_class = CardClass.MAGE.default_hero

    if self._MAX_CARDS_IN_HAND == 0:
      deck1, deck2 = [], []
    else:
      deck1, deck2 = self.generate_decks(self._DECK_SIZE, player1_class, player2_class)

    # NOTE: both players play the same deck
    self.player1 = Player("Agent", deck1, player1_class)
    self.player2 = Player("Opponent", deck1, player2_class)
    # self.player2 = Player("Opponent", deck2, CardClass.WARRIOR.default_hero)

    if self._MAX_CARDS_IN_HAND == 0:
      # allow for coin
      self.player1.max_hand_size = 1
      self.player1._start_hand_size = 1

      self.player2.max_hand_size = 1
      self.player2._start_hand_size = 1
    else:
      self.player1.max_hand_size = self._MAX_CARDS_IN_HAND
      # self.player1._start_hand_size = self._MAX_CARDS_IN_HAND - 1

      self.player2.max_hand_size = self._MAX_CARDS_IN_HAND
      # self.player2._start_hand_size = self._MAX_CARDS_IN_HAND - 1

    self.cheating_opponent = cheating_opponent

    new_game = Game(players=(self.player1, self.player2))
    new_game.MAX_MINIONS_ON_FIELD = self._MAX_CARDS_IN_BOARD
    new_game.start()

    for player in new_game.players:
      player.max_hand_size = self._MAX_CARDS_IN_HAND

    self.player = new_game.players[0]
    self.opponent = new_game.players[1]

    if sort_decks:
      self.player.deck = sorted(self.player.deck, key=lambda x: x.cost)
      self.opponent.deck = sorted(self.opponent.deck, key=lambda x: x.cost)

    if skip_mulligan:
      cards_to_mulligan = self.mulligan_heuristic(self.player1)
      self.player1.choice.choose(*cards_to_mulligan)
      cards_to_mulligan = self.mulligan_heuristic(self.player2)
      self.player2.choice.choose(*cards_to_mulligan)

    if cheating_opponent:
      self.player2.hero._max_health = 100
      self.player2.max_mana = 10
      self.player1.hero._max_health = 1
    elif starting_hp is not None:
      self.player1.hero._max_health = starting_hp
      self.player2.hero._max_health = starting_hp
    self.game = new_game

  @staticmethod
  def mulligan_heuristic(player):
    return [c for c in player.choice.cards if c.cost > 3]

  @staticmethod
  @functools.lru_cache()
  def generate_decks(deck_size, player1_class=CardClass.MAGE,
                     player2_class=CardClass.WARRIOR):
    while True:
      deck1 = utils.random_draft(player1_class, max_mana=5)
      deck2 = utils.random_draft(player2_class, max_mana=5)
      if len(deck1) == deck_size and len(deck2) == deck_size:
        break
    return deck1, deck2

  def terminal(self):
    return self.game.ended

  def actions(self):
    actions = []
    if self.game.current_player.choice:
      for card in self.game.current_player.choice.cards:
        # if not card.is_playable():
        #     continue
        if card.requires_target():
          for target in card.targets:
            actions.append(
              self.Action(card, self.game.current_player.choice.choose, {
                # 'target': target,
                'card': card,
              }, self))
        else:
          actions.append(
            self.Action(card, self.game.current_player.choice.choose, {
              # 'target': None,
              'card': card
            }, self))
    else:
      actions.append(self.Action(None, lambda: None, {}, self))  # PASS

      for card in self.game.current_player.hand:
        if not card.is_playable():
          continue

        # if card.must_choose_one:
        #     for choice in card.choose_cards:
        #         raise NotImplemented()

        elif card.requires_target():
          for target in card.targets:
            actions.append(
              self.Action(card, card.play, {'target': target}, self))
        else:
          actions.append(self.Action(card, card.play, {'target': None}, self))

      for character in self.game.current_player.characters:
        if can_attack(character):
          for enemy_char in character.targets:
            actions.append(
              self.Action(character, character.attack, {'target': enemy_char},
                          self))
      # for action in actions:
      #     action.vector = (self.card_to_bow(action.card), self.card_to_bow(action.params))
    assert len(actions) > 0
    return tuple(actions)

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
      if self.card is None:
        return "PASS"
      target = self.params['target']
      src = '{}({},{})'.format(self.card.data.name, self.card.atk, self.card.health)

      if target is None:
        representation = "PLAY({})".format(src)
      else:
        tar = '{}({},{})'.format(target.data.name, target.atk, target.health)
        representation = 'ATK({},{})'.format(src, tar)

        if self.card.atk >= target.health and target.atk >= self.card.health:
          representation += ' BOTH die'
        elif target.atk >= self.card.health:
          representation += ' ATK dies'
        elif self.card.atk >= target.health:
          representation += ' DEF dies'

      return str(self.card.controller)[0].lower() + representation

    def encode(self):
      state = {
        'card': None if self.card is None else self.hack.card_to_bow(self.card)}
      params_vec = {}
      for k, v in self.params.items():
        params_vec[k] = self.hack.card_to_bow(v)
      state['params'] = params_vec
      return state

  def card_to_vector(self, card):
    SPELL_TYPE = 5
    WEAPON_TYPE = 7
    if card.type == SPELL_TYPE:
      features = [card.cost, 0, 0]
    elif card.type == WEAPON_TYPE:
      features = [card.cost, card.durability, card.atk]
    else:
      features = [card.cost, card.health, card.atk]
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
    ents = [self.entity_to_vec(c) for c in player_hand[:self._MAX_CARDS_IN_HAND]]
    player_hand = np.hstack(ents) if ents else np.array(ents)

    # the player hero itself is not in the board
    player_board = player.characters[1:]
    player_board = list(sorted(player_board, key=lambda x: x.id)) + [None] * self._MAX_CARDS_IN_BOARD

    assert len(player_board) < self._MAX_CARDS_IN_BOARD or not any(player_board[self._MAX_CARDS_IN_BOARD:])

    player_board = np.hstack([self.entity_to_vec(c) for c in player_board[:self._MAX_CARDS_IN_BOARD]])

    player_hero = self.entity_to_vec(player.characters[0])
    player_mana = player.max_mana

    game_state = np.hstack((player_hand, player_board, player_hero, player_mana))
    return game_state

  def observe(self):
    # TODO: encode the player's hand, right now the observation doesnt
    # include your own hand
    # if self.game.current_player == self.player:
    #   active_player, passive_player = self.player, self.opponent
    # else:
    #   active_player, passive_player = self.opponent, self.player

    player_observation = self.observe_player(self.player)
    opponent_observation = self.observe_player(self.opponent)

    if self.game.current_player.controller.name == 'Opponent':
      observation = np.hstack((opponent_observation, player_observation))
    else:
      observation = np.hstack((player_observation, opponent_observation))

    return observation

  def step(self, action):
    action.use()
    observation = self.observe()
    terminal = self.terminal()
    return observation, terminal

  def sudden_death(self):
    self.player.playstate = PlayState.LOSING
    self.game.check_for_end_game()

  @staticmethod
  def encode_to_numerical(k, val):
    if k == "power" and val:
      val = val.data.id
    if isinstance(val, int):  # also bool is int
      pass
    elif val is None:
      val = -1
    elif val is False:
      val = 0
    elif val is True:
      val = 1
    elif isinstance(val, str):
      val = HSsimulation.str_to_vec(val)
    elif isinstance(val, list):
      if len(val) != 0:
        raise Exception("wtf is this list?", val)
      val = 0
    else:
      raise Exception("wtf is this data?", val)
    return val

  @staticmethod
  @utils.disk_cache
  def str_to_vec(val):
    @utils.disk_cache
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

  # TODONT: cards have __hash__ as their ID, which means changes in health wont change the card id
  # @functools.lru_cache(maxsize=20000)
  def card_to_bow(self, card_obj):
    assert card_obj is None or card_obj.data.description == ""

    card_dict = self._card_dict  # .copy()
    # TODO: check all the possible attributes
    for k in card_dict:
      try:
        card_dict[k] = card_obj.__getattribute__(k)
      except AttributeError:
        card_dict[k] = None

    if card_obj is not None:
      card_dict['can_attack'] = card_obj.can_attack()
    else:
      card_dict['can_attack'] = None

    # crash if skipping important data
    if hs_config.DEBUG:
      for k in self._skipped:
        try:
          value = card_obj.__getattribute__(k)
        except AttributeError:
          pass
        else:
          assert self._skipped[k] is None or value in self._skipped[k]

    # card_dict['description'] = ''

    card_lst = []
    for k in card_dict.keys():
      val = card_dict[k]
      # print(k, val)

      val = self.encode_to_numerical(k, val)
      if isinstance(val, int):
        card_lst.append(val)
      elif isinstance(val, np.ndarray):
        raise NotImplementedError(
          "simple environment doesn't support complex cards")
        # card_lst.extend(list(val))
      else:
        raise TypeError()
    assert len(card_lst) == 3
    return np.array(card_lst)

  @staticmethod
  def fast_card_to_bow(card_obj):
    if card_obj is None:
      return np.array((-1, -1, -1))
    else:
      return np.array((card_obj.atk, card_obj.health, can_attack(card_obj)))

  @staticmethod
  def entity_to_vec(entity):
    # crd = self.card_to_bow(entity)
    return HSsimulation.fast_card_to_bow(entity)


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

  def _choose_action(self, state, possible_actions):
    if random.random() < self.epsilon:
      best_action = random.choice(possible_actions)
    else:
      q = [self.getQ(state, self.simulation.action_to_action_id(a)) for a in
           possible_actions]
      maxQ = max(q)
      if maxQ == 0.0:  # FIXME: account for no-action which is always present
        self.qmiss += 1
      else:
        self.qhit += 1
      # choose one of the best actions
      best_action = random.choice(
        [a for q, a in zip(q, possible_actions) if q == maxQ])
    return best_action

  def learn_from_reaction(self, state, action, reward, next_state):
    action_id = self.simulation.action_to_action_id(action)
    state_id = state

    old_q = self.getQ(state_id, action_id)
    change = self.learning_rate * (
      reward + self.gamma * np.max(self.getQ(next_state)) - old_q)
    new_q = old_q + change
    self.setQ(state, action_id, new_q)
    return change
