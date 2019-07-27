import enum

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
