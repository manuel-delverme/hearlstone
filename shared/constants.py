import collections
import enum

import hearthstone.deckstrings

import pysabberstone.python.option


def idx_to_one_hot(index, max_size):
  assert index < max_size
  return tuple([0, ] * index + [1, ] + [0, ] * (max_size - index - 1))


class RewardType(enum.Enum):
  mana_efficency = "mana_efficency"
  hand_adv = "hand_adv"
  life_adv = "life_adv"
  board_adv = "board_adv"
  time_left = "time_left"
  empowerment = "empowerment"
  default = "default"


class SPELLS(enum.IntEnum):
  Polymorph = 77  # transform in a sheep
  Fireball = 315  # 6 damage
  ArcaneExplosion = 447  # 1 damage all
  ArcaneIntellect = 555  # 2 cards
  FrostNova = 587  # freeze all
  # ArcaneMissels = 564  # 3 random damage
  Frostbolt = 662  # 3 damage  + freze
  Flamestrike = 1004  # 4 damage all minions
  MirrorImage = 1084  # summon two minions
  TheCoin = 1746  # + 1 mana


class MINIONS(enum.IntEnum):
  DoomSayer = 138
  NoviceEngineer = 284
  WaterElemental = 395
  GurubashiBerserker = 768
  OgreMagi = 995
  KoboldGeomancer = 672
  AcidSwampOoze = 906
  Archmage = 525
  MirrorImageToken = 968
  PolymorphToken = 796


# DECK1 = r"AAECAf0EAr8D7AcOTZwCuwKLA40EqwS0BMsElgWgBYAGigfjB7wIAA=="
# AAECAR8CWpYNDu0GigfYAeMF0AfZCoEKqAKBAuAEoQLrB9oK8AMA
DECK1 = r"AAECAf0EAr8D7AcOvAiKB4oBlgWgBZwCqwTLBLsC4wdNiwOABo0EAA=="
DECK2 = DECK1

AGENT_ID = 1
OPPONENT_ID = 2


class PlayerTaskType(enum.IntEnum):
  CHOOSE = 0
  CONCEDE = 1
  END_TURN = 2
  HERO_ATTACK = 3
  HERO_POWER = 4
  MINION_ATTACK = 5
  PLAY_CARD = 6


assert all(a == b and repr(a) == repr(b) for a, b in
           zip(PlayerTaskType, pysabberstone.python.option.PlayerTaskType))


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
                                        ['mana_adv', 'hand_adv', 'life_adv', 'n_turns_left', 'board_adv', 'empowerment'])
ACTION_SPACE = 249
STATE_SPACE = 228


class Players(enum.Enum):
  AGENT = 0
  OPPONENT = 1
  LOG = 2


GUI_CARD_WIDTH = 1  # width of box to draw card in
GUI_CARD_HEIGHT = 3

deck = hearthstone.deckstrings.Deck().from_deckstring(DECK1)
Minion = collections.namedtuple('minion', [
  'atk',
  'health',
  'exhausted',
  'card_id',
])

Card = collections.namedtuple('card', [
  'atk',
  'cost',
  'health',
  'card_id',
])

Hero = collections.namedtuple('hero', ['atk', 'health', 'atk_exhausted', 'power_exhausted'])

# 3 is the hand crafted features (atk, health, cost) plus the ~one hot encodings
INACTIVE_CARD_ENCODING_SIZE = len(Card._fields)
ACTIVE_CARD_ENCODING_SIZE = len(Minion._fields)
HERO_ENCODING_SIZE = len(Hero._fields)

DECK_IDS = [card_id for card_id, count in deck.cards for _ in range(count)]
DECK_ID_TO_POSITION = {k: v for (v, k), in reversed(list(zip(enumerate(DECK_IDS))))}

CARD_LOOKUP = {}
REVERSE_CARD_LOOKUP = {}
