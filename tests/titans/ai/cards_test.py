from titans.ai import Ability, Card, Name, Species


def test_card_instantiation():

    # instantiate all possible cards
    for name in Name:

        # create card
        card = Card(name)

        # check cost
        match card.species:
            case Species.WARRIOR:
                assert card.cost == 1
            case Species.BEAST:
                assert card.cost == 2
            case Species.DRAGON:
                assert card.cost == 3
            case Species.TITAN:
                assert card.cost == 4

        # check abilities
        match card.species:
            case Species.BEAST:
                assert Ability.FLASH in card.abilities
            case Species.DRAGON:
                assert (
                    Ability.BOLSTER_STORM in card.abilities
                    or Ability.BOLSTER_FIRE in card.abilities
                    or Ability.BOLSTER_ICE in card.abilities
                    or Ability.BOLSTER_ROCK in card.abilities
                )
