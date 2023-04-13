import numpy as np

from titans.ai import Game, Identity, Name, Network, Player, Trainer


def test__save_history__check_match():

    # initialize
    trainer = Trainer()
    game = Game({"random_state": 271828}).play()
    trainer._save_history(game)

    # check histories match
    for is_winner in [True, False]:

        # get current player
        player: Player = game.players[game.winner]
        if not is_winner:
            player = player.opponent
        identity = player.identity

        # check histories match
        for network in Network:
            for state, choices in game.history[identity][network].items():

                # convert choices from index rep
                counts = np.zeros(len(Name) + 1)
                for choice in choices:
                    counts[choice] += 1

                # make sure matches
                assert (
                    counts == trainer.history[is_winner][network][state]
                ).all()


def test__save_history__check_values():

    # initialize
    trainer = Trainer()
    game = Game()
    state0 = np.array([0., 2.]).tobytes()
    state1 = np.array([0., 1.]).tobytes()
    game.history = {
        identity: {
            network: {
                state0: [0],
                state1: [1, 3, 3],
            }
            for network in Network
        }
        for identity in Identity
    }
    game.winner = Identity.MIKE
    trainer._save_history(game)

    # check
    for is_winner in [True, False]:
        for network in Network:
            history = trainer.history[is_winner][network]
            assert history[state0].sum() == 1
            assert history[state0][0] == 1
            assert history[state1].sum() == 3
            assert history[state1][1] == 1
            assert history[state1][3] == 2


def test_get_Xy():

    # initialize
    default_count = 2 * np.ones(len(Name) + 1)
    trainer = Trainer()
    trainer.history = {
        is_winner: {
            network: {
                np.array([0., 2.]).tobytes():
                    (1 + network) * (not is_winner) * default_count,
                np.array([0., 1.]).tobytes():
                    (1 + network) * is_winner * default_count,
            }
            for network in Network
        }
        for is_winner in [True, False]
    }

    # get Xy
    Xy = trainer.get_Xy()

    # check results
    for network in Network:
        assert (Xy[network][0][0] == [0, 2]).all()
        assert (Xy[network][0][1] == [0, 1]).all()
    assert (Xy[0][1][0] == 0.40).all()
    assert (Xy[0][1][1] == 0.60).all()
    assert (Xy[1][1][0] == 0.25).all()
    assert (Xy[1][1][1] == 0.75).all()
    assert Xy[0][0].shape == (2, 2)
    assert Xy[0][1].shape == (2, len(Name) + 1)


def test_play():
    Trainer().play()
