from titans.ai import Card, Identity, Name, Player


def test___init__():

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


def test_draw():

    # initialize
    cards = [Card(Name.MONK) for _ in range(24)]
    player = Player(Identity.MIKE, cards)

    # draw 4 cards
    player.draw(4)
    assert len(player.deck) == 8
    assert len(player.hand) == 4

    # draw all cards (w/ overdraw)
    player.draw(30)
    assert len(player.deck) == 0
    assert len(player.hand) == 12


def test_shuffle():

    # initialize
    cards = [Card(Name.WIZARD) for _ in range(24)]
    player = Player(Identity.MIKE, cards)

    # draw, then shuffle
    player.draw(4)
    player.shuffle()
    assert len(player.deck) == 12
    assert len(player.hand) == 0


def test_state():

    # initialize
    player = Player(Identity.MIKE, [])
    player.discard.extend([Card(Name.MONK) for _ in range(2)])
    player.play.append(Card(Name.AURORA_DRACO))
    player.deck.extend([Card(Name.FINAL_JUDGMENT), Card(Name.GHOST)])
    player.hand.append(Card(Name.GHOST))

    # check private state
    private_state = player.state(public=False)
    assert private_state[Name.MONK] == 2
    assert private_state[len(Name) + Name.AURORA_DRACO] == 1
    assert private_state[2 * len(Name) + Name.FINAL_JUDGMENT] == 1
    assert private_state[2 * len(Name) + Name.GHOST] == 1
    assert private_state[3 * len(Name) + Name.GHOST] == 1
    assert private_state[-4] == 2
    assert private_state[-3] == 1
    assert private_state[-2] == 2
    assert private_state[-1] == 1

    # check public state
    public_state = player.state(public=True)
    assert public_state[Name.MONK] == 2
    assert public_state[len(Name) + Name.AURORA_DRACO] == 1
    assert public_state[2 * len(Name) + Name.FINAL_JUDGMENT] == 1
    assert public_state[2 * len(Name) + Name.GHOST] == 2
    assert public_state[-4] == 2
    assert public_state[-3] == 1
    assert public_state[-2] == 2
    assert public_state[-1] == 1
    assert len(public_state) + len(Name) == len(private_state)
