import numpy as np

from titans.ai import Action, Game, Identity, Name, Player


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


def test__play_age():

    # set strategy to always choose to play Monk, and alway awaken Nikolai
    strategies = [
        np.zeros((
            Player._get_global_state_size(),
            len(Name) + 1,
        ))
        for _ in Action
    ]
    strategies[Action.AWAKEN][:, Name.NIKOLAI_THE_CURSED] = 1
    strategies[Action.PLAY][:, Name.MONK] = 1

    # draw cards, play age
    game = Game({"strategies": strategies})
    for player in game.players:
        player.draw_cards(6)
    game._play_age()

    # ensure correct cards played and awakened
    for player in game.players:
        assert player.play_zone[0].name == Name.MONK
        assert player.discard_zone[0].name == Name.NIKOLAI_THE_CURSED


def test__play_turn():

    # set strategies
    strategies = [
        np.zeros((
            Player._get_global_state_size(),
            len(Name) + 1,
        ))
        for _ in Action
    ]
    strategies[Action.AWAKEN][:, Name.NIKOLAI_THE_CURSED] = 1
    strategies[Action.AWAKEN][:, Name.WINDS_HOWL] = 2
    strategies[Action.AWAKEN][:, Name.FROSTBREATH] = 3
    strategies[Action.PLAY][:, Name.MONK] = 2
    strategies[Action.PLAY][:, len(Name)] = 1

    # draw cards, play age
    game = Game(
        {"strategies": strategies, "random_state": 271828},
        {"random_state": 42},
    )
    game._play_turn()

    # ensure correct cards played and awakened
    assert all([
        card.name == Name.MONK
        for card in game.players[0].play_zone
    ])
    assert game.players[0].discard_zone[0].name == Name.NIKOLAI_THE_CURSED
    assert game.players[0].discard_zone[1].name == Name.WINDS_HOWL
    assert game.players[0].discard_zone[2].name == Name.FROSTBREATH
    assert not all([
        c0.name == c1.name
        for c0, c1 in zip(
            game.players[0].discard_zone,
            game.players[1].discard_zone,
        )
    ])

    # play two more turns, check temple count
    game._play_turn()
    game._play_turn()
    assert (
        (game.players[0].temples < 3)
        or (game.players[1].temples < 3)
    )

    # check history
    choices = list(game.history[Identity.MIKE][Action.AWAKEN].values())
    assert len(choices[0]) == 1
    assert choices[0][0] == Name.NIKOLAI_THE_CURSED
    assert len(choices[1]) == 1
    assert choices[1][0] == Name.WINDS_HOWL
    assert len(choices[2]) == 1
    assert choices[2][0] == Name.FROSTBREATH


def test_play():
    game = Game().play()
    assert game.players[game.winner].temples > 0
    assert game.players[game.winner].opponent.temples == 0
