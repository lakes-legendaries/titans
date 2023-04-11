from titans.ai import Card, Identity, Name, Player


def test_deck_creation():

    # create sample cards
    cards = []
    cards.extend([Card(Name.MONK) for _ in range(20)])
    cards.extend([Card(Name.WIZARD) for _ in range(10)])
    cards.extend([Card(Name.GHOST) for _ in range(100)])

    # create players
    players = [Player(identity, cards) for identity in Identity]

    # check each starting deck has the right cards
    assert len(players[0].deck) == len(players[1].deck) == 15
    for player in players:
        assert len([
            card
            for card in player.deck
            if card.name == Name.MONK
        ]) == 10
        assert len([
            card
            for card in player.deck
            if card.name == Name.WIZARD
        ]) == 5

    # ensure each card was only dealt to one player
    for card0 in players[0].deck:
        for card1 in players[1].deck:
            assert not (card0 is card1)
