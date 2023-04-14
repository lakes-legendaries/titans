from titans.ai import Identity, NUM_FEATURES, Player


def test_num_features():
    players = [Player(identity, []) for identity in Identity]
    players[0].handshake(players[1])
    assert len(players[0].get_state()) == NUM_FEATURES
