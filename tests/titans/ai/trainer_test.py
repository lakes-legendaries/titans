import numpy as np

from titans.ai import (
    Action,
    Game,
    Identity,
    NUM_CHOICES,
    Player,
    Trainer,
)


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
        for action in Action:
            for state, choices in game.history[identity][action].items():

                # convert choices from index rep
                counts = np.zeros(NUM_CHOICES)
                for choice in choices:
                    counts[choice] += 1

                # make sure matches
                assert (
                    counts == trainer.history[is_winner][action][state]
                ).all()


def test__save_history__check_values():

    # initialize
    trainer = Trainer()
    game = Game()
    state0 = np.array([0., 2.]).tobytes()
    state1 = np.array([0., 1.]).tobytes()
    game.history = {
        identity: {
            action: {
                state0: [0],
                state1: [1, 3, 3],
            }
            for action in Action
        }
        for identity in Identity
    }
    game.winner = Identity.MIKE
    trainer._save_history(game)

    # check
    for is_winner in [True, False]:
        for action in Action:
            history = trainer.history[is_winner][action]
            assert history[state0].sum() == 1
            assert history[state0][0] == 1
            assert history[state1].sum() == 3
            assert history[state1][1] == 1
            assert history[state1][3] == 2


def test_get_Xy():

    # initialize
    default_count = 2 * np.ones(NUM_CHOICES)
    trainer = Trainer()
    trainer.history = {
        is_winner: {
            action: {
                np.array([0., 2.]).tobytes():
                    (1 + action.value) * (not is_winner) * default_count,
                np.array([0., 1.]).tobytes():
                    (1 + action.value) * is_winner * default_count,
            }
            for action in Action
        }
        for is_winner in [True, False]
    }

    # get Xy
    Xy = trainer.get_Xy()

    # check results
    for action in Action:
        assert (Xy[action][0][0] == [0, 2]).all()
        assert (Xy[action][0][1] == [0, 1]).all()
    assert (Xy[Action(0)][1][0] == 0.40).all()
    assert (Xy[Action(0)][1][1] == 0.60).all()
    assert (Xy[Action(1)][1][0] == 0.25).all()
    assert (Xy[Action(1)][1][1] == 0.75).all()
    assert Xy[Action(0)][0].shape == (2, 2)
    assert Xy[Action(0)][1].shape == (2, NUM_CHOICES)


def test_play():
    Trainer().play()


test_play()
