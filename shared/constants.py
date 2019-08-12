import collections
import enum

import hearthstone.deckstrings


def idx_to_one_hot(index, max_size):
  assert index < max_size
  return tuple([0, ] * index + [1, ] + [0, ] * (max_size - index - 1))


class SPELLS(enum.IntEnum):
  Polymorph = 77  # transform in a sheep
  Fireball = 315  # 6 damage
  ArcaneExplosion = 447  # 1 damage all
  ArcaneIntellect = 555  # 2 cards
  FrostNova = 587  # freeze all
  ArcaneMissels = 564  # 3 random damage
  Frostbolt = 662  # 3 damage  + freze
  Flamestrike = 1004  # 4 damage all minions
  MirrorImage = 1084  # summon two minions
  TheCoin = 1746  # + 1 manA


class MINIONS(enum.IntEnum):
  NoviceEngineer = 284
  WaterElemental = 395
  GurubashiBerserker = 768
  OgreMagi = 995
  KoboldGeomancer = 672
  AcidSwampOoze = 906
  Archmage = 525
  MirrorImageToken = 968
  PolymorphToken = 796


DECK1 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="
DECK2 = DECK1

AGENT_ID = 1
OPPONENT_ID = 2


# reverse enumerating allows for the first of two indices to overwrite the second, e.g. the idxes are 1..3..5 not 2..4..6

# DECK_DUPLICATES = {card_id for card_id, count in deck.cards if count == 1}


class PlayerTaskType(enum.IntEnum):
  CHOOSE = 0
  CONCEDE = 1
  END_TURN = 2
  HERO_ATTACK = 3
  HERO_POWER = 4
  MINION_ATTACK = 5
  PLAY_CARD = 6


class BoardPosition(enum.IntEnum):
  RightMost = -1
  Hero = 0
  B1 = 1
  B2 = 2
  B3 = 3
  B4 = 4
  B5 = 5
  B6 = 6
  B7 = 7
  oHero = 0 + 8
  oB1 = 1 + 8
  oB2 = 2 + 8
  oB3 = 3 + 8
  oB4 = 4 + 8
  oB5 = 5 + 8
  oB6 = 6 + 8
  oB7 = 7 + 8


class HandPosition(enum.IntEnum):
  H1 = 0
  H2 = 1
  H3 = 2
  H4 = 3
  H5 = 4
  H6 = 5
  H7 = 6
  H8 = 7
  H9 = 8
  HA = 9


BoardPosition.__call__ = lambda x: x
HandPosition.__call__ = lambda x: x
PlayerTaskType.__call__ = lambda x: x

GameStatistics = collections.namedtuple('GameStatistics',
                                        ['mana_adv', 'hand_adv', 'draw_adv', 'life_adv', 'n_turns_left', 'minion_adv'])
ACTION_SPACE = 249
# STATE_SPACE = 635
STATE_SPACE = 698


class Players(enum.Enum):
  AGENT = 0
  OPPONENT = 1
  LOG = 2


GUI_CARD_WIDTH = 1  # width of box to draw card in
GUI_CARD_HEIGHT = 3

deck = hearthstone.deckstrings.Deck().from_deckstring(DECK1)
# extra_spells = [SPELLS.TheCoin]
# extra_minions = [MINIONS.MirrorImageToken, MINIONS.PolymorphToken]

# SPELL_IDS = set(SPELLS)
# MINION_IDS = set(MINIONS)
_ONE_HOT_LENGTH = max(len(MINIONS), len(SPELLS))  # largest set of different cards

# minion_onehots = []
# minion_name = str(MINIONS(idx)).split('.')[1]
# cards_onehots = minion_onehots
# spell_name = str(SPELLS(idx)).split('.')[1]

cards_onehots = [str(c).split('.')[1] for c in MINIONS]
Minion = collections.namedtuple('minion', ['atk', 'health', 'exhausted'] + cards_onehots.copy())

for order_idx, spell in enumerate(SPELLS):
  spell = str(spell).split('.')[1]

  if order_idx < len(cards_onehots):
    cards_onehots[order_idx] += f"_{spell}"
  else:
    cards_onehots.append(spell)

Card = collections.namedtuple('card', ['atk', 'health', 'cost', ] + cards_onehots.copy())
Hero = collections.namedtuple('hero', ['atk', 'health', 'atk_exhausted', 'power_exhausted'])

# assert all(card_id in set(MINIONS) or card_id in set(SPELLS) for card_id, count in deck.cards)
# assert deck.heroes == [637, ]

# SPELL_IDS.update(extra_spells)
# MINION_IDS.update(extra_minions)

# assert not {card_id for card_id, count in deck.cards} - {*MINION_IDS, *SPELL_IDS}, str(
#     {card_id for card_id, count in deck.cards} - {*MINION_IDS, *SPELL_IDS}) + ' was not handled'

# 3 is the hand crafted features (atk, health, cost) plus the ~one hot encodings
INACTIVE_CARD_ENCODING_SIZE = len(Card._fields)
ACTIVE_CARD_ENCODING_SIZE = len(Minion._fields)
HERO_ENCODING_SIZE = len(Hero._fields)

INACTIVE_CARDS_ONE_HOT = {k: idx_to_one_hot(idx, _ONE_HOT_LENGTH) for idx, k in enumerate(MINIONS)}
INACTIVE_CARDS_ONE_HOT.update({k: idx_to_one_hot(idx, _ONE_HOT_LENGTH) for idx, k in enumerate(SPELLS)})

ACTIVE_CARDS_ONE_HOT = {k: idx_to_one_hot(idx, len(MINIONS)) for idx, k in enumerate(MINIONS)}

# DECK_POSITION_LOOKUP = {k: idx for idx, k in enumerate((*MINIONS, *SPELLS))}
# MINIONS_ORDER = [MINIONS(card_id) for card_id, count in deck.cards if card_id in MINION_IDS]
# SPELLS_ORDER = [SPELLS(card_id) for card_id, count in deck.cards if card_id in SPELL_IDS]
DECK_IDS = [card_id for card_id, count in deck.cards for _ in range(count)]
DECK_ID_TO_POSITION = {k: v for (v, k), in reversed(list(zip(enumerate(DECK_IDS))))}

REVERSE_SPELL_LOOKUP = {idx_to_one_hot(idx, _ONE_HOT_LENGTH): k for idx, k in enumerate(SPELLS)}
REVERSE_MINION_LOOKUP = {idx_to_one_hot(idx, _ONE_HOT_LENGTH): k for idx, k in enumerate(MINIONS)}
