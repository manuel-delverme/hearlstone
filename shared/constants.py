import collections
import enum
from enum import IntEnum

AGENT_ID = 1
OPPONENT_ID = 2


class SPELLS(enum.IntEnum):
  Polymorph = 77  # transform in a sheep
  Fireball = 315  # 6 damage
  ArcaneExplosion = 447  # 1 damage all
  ArcaneIntellect = 555  # 2 cards
  FrostNova = 587  # freeze all TODO: implement
  ArcaneMissels = 564  # 3 random damage
  Frostbolt = 662  # 3 damage  + freze
  Flamestrike = 1004  # 4 damage all minions
  MirrorImage = 1084  # summon two minions
  TheCoin = 1746  # + 1 mana


class PlayerTaskType(IntEnum):
  CHOOSE = 0
  CONCEDE = 1
  END_TURN = 2
  HERO_ATTACK = 3
  HERO_POWER = 4
  MINION_ATTACK = 5
  PLAY_CARD = 6


class BoardPosition(IntEnum):
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


class HandPosition(IntEnum):
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
_ACTION_SPACE = 249
_STATE_SPACE = 92  # state space includes card_index

Minion = collections.namedtuple('minion', ['atk', 'health', 'exhausted'])
Card = collections.namedtuple('card', ['id', 'atk', 'health', 'cost'])
Hero = collections.namedtuple('hero', ['atk', 'health', 'atk_exhausted', 'power_exhausted'])