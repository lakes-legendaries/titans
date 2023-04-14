"""Common enums"""

from enum import auto, Enum


class Ability(Enum):
    """Card Abilities"""
    BOLSTER_FIRE = 0
    BOLSTER_ICE = auto()
    BOLSTER_RIVALS = auto()
    BOLSTER_ROCK = auto()
    BOLSTER_STORM = auto()
    DISCARD = auto()
    DRAW = auto()
    ENERGY = auto()
    ENERGY_ARC = auto()
    FLASH = auto()
    HAUNT = auto()
    PROTECT = auto()
    PURIFY = auto()
    SACRIFICE = auto()
    SUBSTITUTE = auto()
    SUBVERT_CAVE_IN = auto()
    SUBVERT_HARMLESS = auto()
    SUBVERT_MINDLESS = auto()
    SUBVERT_TRAITOROUS = auto()
    SUMMON = auto()


class Action(Enum):
    """Various actions (one for each strategy)"""
    AWAKEN = 0
    PLAY = auto()


class Element(Enum):
    """Card Elements"""
    FOREST = 0
    DESERT = auto()
    STORM = auto()
    FIRE = auto()
    ICE = auto()
    ROCK = auto()


class Identity(Enum):
    """Player names"""
    MIKE = 0
    BRYAN = auto()


class Name(Enum):
    """Card Names"""

    # Forest
    MONK = 0
    WIZARD = auto()
    TRAVELER = auto()

    # Desert
    GHOST = auto()

    # Storm
    NIKOLAI_THE_CURSED = auto()
    WINDS_HOWL = auto()
    AURORA_DRACO = auto()
    MADNESS_OF_A_THOUSAND_STARS = auto()

    # Fire
    ZODIAC_THE_ETERNAL = auto()
    LIVING_VOLCANO = auto()
    SMOLDERING_DRAGON = auto()
    FINAL_JUDGMENT = auto()

    # Ice
    JACE_WINTERS_FIRSTBORN = auto()
    RETURN_OF_THE_FROST_GIANTS = auto()
    FROSTBREATH = auto()
    HELL_FROZEN_OVER = auto()

    # Rock
    AKARI_TIMELESS_FIGHTER = auto()
    SPINE_SPLITTER = auto()
    CAVERNS_DEFENDER = auto()
    WHAT_LIES_BELOW = auto()


class Species(Enum):
    """Card Species"""
    DWELLER = 0
    WARRIOR = auto()
    BEAST = auto()
    DRAGON = auto()
    TITAN = auto()


class Zone(Enum):
    """Different zones a player's cards can be in"""
    DECK = 0
    DISCARD = 1
    HAND = 2
    PLAY = 3
