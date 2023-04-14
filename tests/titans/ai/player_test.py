import numpy as np

from titans.ai import (
    Action,
    Card,
    Identity,
    Name,
    NUM_CHOICES,
    Player,
    Strategy,
)


def test___init__():

    # create sample cards
    cards = []
    cards.extend([Card(Name.MONK) for _ in range(20)])
    cards.extend([Card(Name.WIZARD) for _ in range(10)])
    cards.extend([Card(Name.GHOST) for _ in range(100)])

    # create players
    players = [Player(identity, cards) for identity in Identity]
    players[0].handshake(players[1])

    # check each starting deck has the right cards
    assert len(players[0].deck_zone) == len(players[1].deck_zone) == 15
    for player in players:
        assert len([
            card
            for card in player.deck_zone
            if card.name == Name.MONK
        ]) == 10
        assert len([
            card
            for card in player.deck_zone
            if card.name == Name.WIZARD
        ]) == 5

    # ensure each card was only dealt to one player
    for card0 in players[0].deck_zone:
        for card1 in players[1].deck_zone:
            assert not (card0 is card1)

    # check ritual piles have the right cards
    assert players[0].ritual_piles is players[1].ritual_piles
    assert len(players[0].ritual_piles) == 100
    assert all([card.name == Name.GHOST for card in players[0].ritual_piles])


def test__get_global_state():

    # initialize
    players = [
        Player(identity, [], random_state=random_state)
        for identity, random_state in zip(Identity, [42, 271828])
    ]
    for player in players:
        player.deck_zone.extend([Card(name) for name in Name])
        player.shuffle_cards()
        player.draw_cards(6)
    players[0].handshake(players[1])

    # ensure hands aren't equal (which is a check on shuffle_cards)
    assert any([
        c0.name != c1.name
        for c0, c1 in zip(players[0].hand_zone, players[1].hand_zone)
    ])

    # toss out a few cards from the deck, so that public states don't match
    for player in players:
        for _ in range(5):
            player.deck_zone.pop()

    # get states
    public_states = [player.get_state(public=True) for player in players]
    private_states = [player.get_state(public=False) for player in players]

    # check states don't match
    assert (public_states[0] != public_states[1]).any()
    assert (private_states[0] != private_states[1]).any()

    # check global states
    assert (players[0]._get_global_state() == np.concatenate((
        private_states[0],
        public_states[1],
    ))).all()
    assert (players[1]._get_global_state() == np.concatenate((
        private_states[1],
        public_states[0],
    ))).all()


def test_awaken_card():

    # initialize
    cards = []
    cards.extend([Card(Name.GHOST) for _ in range(12)])
    cards.append(Card(Name.AURORA_DRACO))
    players = [Player(identity, cards) for identity in Identity]
    players[0].handshake(players[1])

    # try to awaken with zero energy
    players[0].awaken_card()
    assert len(players[0].discard_zone) == 0

    # set strategy to choose aurora draco 1st, ghosts 2nd
    class ModifiedStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.AURORA_DRACO.value] = 2
            pred[Name.GHOST.value] = 1
            return pred
    players[0].strategies[Action.AWAKEN] = ModifiedStrategy()

    # add energy, check we buy what we can
    players[0].play_zone.append(Card(Name.MONK))
    players[0].awaken_card()
    assert len(players[0].discard_zone) == 1
    assert players[0].discard_zone[0].name == Name.GHOST

    # add energy, see how this changes, check we buy what we can
    players[0].play_zone.append(Card(Name.TRAVELER))
    players[0].awaken_card()
    assert len(players[0].discard_zone) == 2
    assert players[0].discard_zone[-1].name == Name.AURORA_DRACO

    # change strategy to buy ghost
    class ModifiedStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.GHOST.value] = 2
            pred[Name.AURORA_DRACO.value] = 1
            return pred
    players[0].strategies[Action.AWAKEN] = ModifiedStrategy()
    players[0].awaken_card()
    assert len(players[0].discard_zone) == 3
    assert players[0].discard_zone[-1].name == Name.GHOST

    # change strategy to not buy
    class ModifiedStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[-1] = 1
            return pred
    players[0].strategies[Action.AWAKEN] = ModifiedStrategy()
    players[0].awaken_card()
    assert len(players[0].discard_zone) == 3


def test_battle_opponent():

    # initialize
    players = [Player(identity, []) for identity in Identity]
    players[0].handshake(players[1])

    # battle: tied power
    assert players[0].battle_opponent() is None
    assert players[0].temples == players[1].temples == 3

    # battle: power off by one
    players[0].play_zone.append(Card(Name.WIZARD))
    assert players[0].battle_opponent() is None
    assert players[0].temples == players[1].temples == 3

    # battle: power off by two
    players[0].play_zone.append(Card(Name.WIZARD))
    assert players[0].battle_opponent() == players[0].identity
    assert players[0].temples == 3
    assert players[1].temples == 2

    # battle: power off by three
    players[0].play_zone.append(Card(Name.WIZARD))
    assert players[0].battle_opponent() == players[0].identity
    assert players[0].temples == 3
    assert players[1].temples == 1

    # battle: comeback
    players[1].play_zone.extend([Card(Name.WIZARD) for _ in range(5)])
    assert players[0].battle_opponent() == players[1].identity
    assert players[0].temples == 2
    assert players[1].temples == 2


def test_draw_cards():

    # initialize
    cards = [Card(Name.MONK) for _ in range(24)]
    player = Player(Identity.MIKE, cards)

    # draw 4 cards
    player.draw_cards(4)
    assert len(player.deck_zone) == 8
    assert len(player.hand_zone) == 4

    # draw all cards (w/ overdraw)
    player.draw_cards(30)
    assert len(player.deck_zone) == 0
    assert len(player.hand_zone) == 12


def test_freeze_state():
    players = [Player(identity, []) for identity in Identity]
    players[0].handshake(players[1])
    players[0].freeze_state()
    players[0].discard_zone.append(Card(Name.GHOST))
    assert (players[0]._get_global_state() == 0).all()
    players[0].unfreeze_state()
    assert players[0]._get_global_state()[Name.GHOST.value] == 1


def test_get_energy():
    player = Player(Identity.MIKE, [])
    assert player.get_energy() == 0
    player.play_zone.append(Card(Name.MONK))
    assert player.get_energy() == 1
    player.play_zone.append(Card(Name.TRAVELER))
    assert player.get_energy() == 3


def test_get_power():
    player = Player(Identity.MIKE, [])
    assert player.get_power() == 0
    player.play_zone.append(Card(Name.MONK))
    assert player.get_power() == 0
    player.play_zone.append(Card(Name.TRAVELER))
    assert player.get_power() == 1
    player.play_zone.append(Card(Name.WIZARD))
    assert player.get_power() == 2
    player.play_zone.append(Card(Name.WIZARD))
    assert player.get_power() == 3


def test_get_state():

    # initialize
    player = Player(Identity.MIKE, [])
    player.discard_zone.extend([Card(Name.MONK) for _ in range(2)])
    player.play_zone.append(Card(Name.AURORA_DRACO))
    player.deck_zone.extend([Card(Name.FINAL_JUDGMENT), Card(Name.GHOST)])
    player.hand_zone.append(Card(Name.GHOST))

    # check private state
    private_state = player.get_state(public=False)
    assert private_state[Name.MONK.value] == 2
    assert private_state[len(Name) + Name.AURORA_DRACO.value] == 1
    assert private_state[2 * len(Name) + Name.FINAL_JUDGMENT.value] == 1
    assert private_state[2 * len(Name) + Name.GHOST.value] == 1
    assert private_state[3 * len(Name) + Name.GHOST.value] == 1
    assert private_state[-4] == 2
    assert private_state[-3] == 1
    assert private_state[-2] == 2
    assert private_state[-1] == 1

    # check public state
    public_state = player.get_state(public=True)
    assert public_state[Name.MONK.value] == 2
    assert public_state[len(Name) + Name.AURORA_DRACO.value] == 1
    assert public_state[2 * len(Name) + Name.FINAL_JUDGMENT.value] == 1
    assert public_state[2 * len(Name) + Name.GHOST.value] == 2
    assert public_state[-4] == 2
    assert public_state[-3] == 1
    assert public_state[-2] == 2
    assert public_state[-1] == 1
    assert len(public_state) + len(Name) == len(private_state)


def test_play_cards():

    # initialize
    cards = []
    cards.extend([Card(Name.MONK) for _ in range(16)])
    cards.extend([Card(Name.WIZARD) for _ in range(8)])
    players = [
        Player(identity, cards, random_state=random_state)
        for identity, random_state in zip(Identity, [42, 271828])
    ]
    players[0].handshake(players[1])
    players[0].shuffle_cards()
    players[0].draw_cards(6)
    players[0].hand_zone.append(Card(Name.AURORA_DRACO))

    # set strategy to always choose aurora draco
    class ModifiedStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.AURORA_DRACO.value] = 1
            return pred
    players[0].strategies[Action.PLAY] = ModifiedStrategy()

    # play card, check aurora draco was played
    players[0].play_cards()
    assert players[0].play_zone[0].name == Name.AURORA_DRACO

    # change strategy to play top card of the deck
    class ModifiedStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[-1] = 1
            return pred
    players[0].strategies[Action.PLAY] = ModifiedStrategy()
    players[0].play_cards()
    assert len(players[0].deck_zone) == 5


def test_shuffle_cards():

    # initialize
    cards = [Card(Name.WIZARD) for _ in range(24)]
    player = Player(Identity.MIKE, cards)

    # draw, then shuffle
    player.draw_cards(4)
    player.shuffle_cards()
    assert len(player.deck_zone) == 12
    assert len(player.hand_zone) == 0

    # show cards in discard go to deck
    player.discard_zone.append(Card(Name.GHOST))
    player.shuffle_cards()
    assert len(player.deck_zone) == 13
    assert sum([card.name == Name.GHOST for card in player.deck_zone]) == 1
