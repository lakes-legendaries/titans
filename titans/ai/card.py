"""Cards"""

from titans.ai.enum import Ability, Element, Name, Species


class Card:
    """Card, with all properties

    Parameters
    ----------
    name: Name
        Name of card

    Attributes
    ----------
    abilities: dict[Ability, int]
        card abilities (count of each ability)
    cost: int
        amount of energy required to awaken
    element: Element
        card element
    name: Name
        card name
    power: int
        card power
    species: Species
        card species
    """
    def __init__(
        self,
        name: Name,
        /,
    ):
        # check parameters
        if not isinstance(name, Name):
            raise ValueError(f"{name} is not a member of titans.ai.enum.Name")

        # save name
        self.name = name

        # assign properties
        match self.name:

            # forest cards
            case Name.MONK:
                self.abilities = {
                    Ability.ENERGY: 1
                }
                self.cost = 0
                self.element = Element.FOREST
                self.power = 0
                self.species = Species.DWELLER
            case Name.WIZARD:
                self.abilities = {}
                self.cost = 0
                self.element = Element.FOREST
                self.power = 1
                self.species = Species.DWELLER
            case Name.TRAVELER:
                self.abilities = {
                    Ability.ENERGY: 2,
                }
                self.cost = 2
                self.element = Element.FOREST
                self.power = 1
                self.species = Species.DWELLER

            # desert cards
            case Name.GHOST:
                self.abilities = {}
                self.cost = 1
                self.element = Element.DESERT
                self.power = -1
                self.species = Species.DWELLER

            # storm cards
            case Name.NIKOLAI_THE_CURSED:
                self.abilities = {
                    Ability.SUMMON: 1,
                }
                self.cost = 1
                self.element = Element.STORM
                self.power = 0
                self.species = Species.WARRIOR
            case Name.WINDS_HOWL:
                self.abilities = {
                    Ability.FLASH: 2,
                }
                self.cost = 2
                self.element = Element.STORM
                self.power = -1
                self.species = Species.BEAST
            case Name.AURORA_DRACO:
                self.abilities = {
                    Ability.HAUNT: 2,
                    Ability.BOLSTER_FIRE: 1,
                }
                self.cost = 3
                self.element = Element.STORM
                self.power = 1
                self.species = Species.DRAGON
            case Name.MADNESS_OF_A_THOUSAND_STARS:
                self.abilities = {
                    Ability.ENERGY_ARC: 4,
                }
                self.cost = 4
                self.element = Element.STORM
                self.power = 1
                self.species = Species.TITAN

            # fire cards
            case Name.ZODIAC_THE_ETERNAL:
                self.abilities = {
                    Ability.PURIFY: 1,
                }
                self.cost = 1
                self.element = Element.FIRE
                self.power = 2
                self.species = Species.WARRIOR
            case Name.LIVING_VOLCANO:
                self.abilities = {
                    Ability.FLASH: 1,
                    Ability.DISCARD: 2,
                }
                self.cost = 2
                self.element = Element.FIRE
                self.power = 0
                self.species = Species.BEAST
            case Name.SMOLDERING_DRAGON:
                self.abilities = {
                    Ability.PROTECT: 1,
                    Ability.BOLSTER_ICE: 1,
                }
                self.cost = 3
                self.element = Element.FIRE
                self.power = 3
                self.species = Species.DRAGON
            case Name.FINAL_JUDGMENT:
                self.abilities = {
                    Ability.DISCARD: 3000,
                }
                self.cost = 4
                self.element = Element.FIRE
                self.power = 2
                self.species = Species.TITAN

            # ice cards
            case Name.JACE_WINTERS_FIRSTBORN:
                self.abilities = {
                    Ability.SUBVERT_HARMLESS: 1,
                }
                self.cost = 1
                self.element = Element.ICE
                self.power = 1
                self.species = Species.WARRIOR
            case Name.RETURN_OF_THE_FROST_GIANTS:
                self.abilities = {
                    Ability.FLASH: 1,
                    Ability.SUBSTITUTE: 1,
                }
                self.cost = 2
                self.element = Element.ICE
                self.power = 0
                self.species = Species.BEAST
            case Name.FROSTBREATH:
                self.abilities = {
                    Ability.SUBVERT_MINDLESS: 1,
                    Ability.BOLSTER_ROCK: 1,
                }
                self.cost = 3
                self.element = Element.ICE
                self.power = 2
                self.species = Species.DRAGON
            case Name.HELL_FROZEN_OVER:
                self.abilities = {
                    Ability.SUBVERT_TRAITOROUS: 1,
                }
                self.cost = 4
                self.element = Element.ICE
                self.power = 1
                self.species = Species.TITAN

            # rock cards
            case Name.AKARI_TIMELESS_FIGHTER:
                self.abilities = {
                    Ability.DRAW: 2,
                }
                self.cost = 1
                self.element = Element.ROCK
                self.power = 2
                self.species = Species.WARRIOR
            case Name.SPINE_SPLITTER:
                self.abilities = {
                    Ability.FLASH: 1,
                    Ability.SACRIFICE: 1,
                }
                self.cost = 2
                self.element = Element.ROCK
                self.power = 0
                self.species = Species.BEAST
            case Name.CAVERNS_DEFENDER:
                self.abilities = {
                    Ability.SUBVERT_CAVE_IN: 1,
                    Ability.BOLSTER_STORM: 1,
                }
                self.cost = 3
                self.element = Element.ROCK
                self.power = 3
                self.species = Species.DRAGON
            case Name.WHAT_LIES_BELOW:
                self.abilities = {
                    Ability.BOLSTER_RIVALS: 1,
                }
                self.cost = 4
                self.element = Element.ROCK
                self.power = 0
                self.species = Species.TITAN

            # error
            case _:
                raise NotImplementedError(
                    f"Missing properties for {name.name}"
                )

    def __repr__(self) -> str:
        return f"{self.name.name} Card"
