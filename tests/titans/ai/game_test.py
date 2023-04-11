import numpy as np

from titans.ai import Game, Name, Network, Player


def test___init__():

    # test basic construction
    game = Game()
    assert len(game.players) == 2
    assert len(game.players[0].ritual_piles) == 4 * 4 * 4 + 12 + 8

    # test passing one arg
    dummy_strategy = np.zeros(7)  # won't actually work for playing
    game = Game({"strategies": [None, dummy_strategy]})
    assert (game.players[0].strategies[1] == dummy_strategy).all()
    assert (game.players[1].strategies[1] == dummy_strategy).all()

    # test passing two args
    dummy_strategy = np.zeros(7)  # won't actually work for playing
    game = Game(
        {"strategies": [None, dummy_strategy]},
        {"strategies": [dummy_strategy, None]},
    )
    assert game.players[0].strategies[0] is None
    assert game.players[1].strategies[1] is None
    assert (game.players[0].strategies[1] == dummy_strategy).all()
    assert (game.players[1].strategies[0] == dummy_strategy).all()


def test_play_age():

    # set strategy to always choose to play Monk, and alway awaken Nikolai
    strategies = [
        np.zeros((
            Player._get_global_state_size(),
            len(Name) + 1,
        ))
        for _ in Network
    ]
    strategies[Network.AWAKEN][:, Name.NIKOLAI_THE_CURSED] = 1
    strategies[Network.PLAY][:, Name.MONK] = 1

    # draw cards, play age
    game = Game({"strategies": strategies})
    for player in game.players:
        player.draw_cards(6)
    game.play_age()

    # ensure correct cards played and awakened
    for player in game.players:
        assert player.play_zone[0].name == Name.MONK
        assert player.discard_zone[0].name == Name.NIKOLAI_THE_CURSED
