import numpy as np

from titans.ai import (
    Action,
    Card,
    Game,
    Identity,
    Name,
    NUM_CHOICES,
    Player,
    Strategy,
    Trainer,
    Zone,
)


def test__get_Xy():

    # initialize
    trainer = Trainer()
    trainer.history = [{
        np.array([0., 1.]).tobytes(): {
            Action.AWAKEN: {
                True: [0, 0, 0, 0],
            },
            Action.PLAY: {
                False: [0],
            },
        },
        np.array([0., 2.]).tobytes(): {
            Action.AWAKEN: {
                True: [0, 0, 0, 0],
                False: [0, 0],
            },
            Action.PLAY: {
                True: [2],
                False: [2],
            },
        },
    }]

    # get Xy
    Xy = trainer._get_Xy()

    # check results
    for action in Action:
        assert (Xy[action][0][0] == [0, 1]).all()
        assert (Xy[action][0][1] == [0, 2]).all()
    assert Xy[Action.AWAKEN][1][0][0] == 1
    assert Xy[Action.AWAKEN][1][1][0] == 4 / 6
    assert Xy[Action.PLAY][1][0][0] == 0
    assert Xy[Action.PLAY][1][1][2] == 0.5
    assert np.isnan(Xy[Action.PLAY][1][1][5])
    assert Xy[Action(0)][0].shape == (2, 2)
    assert Xy[Action(0)][1].shape == (2, NUM_CHOICES)


def test__parallel_step():

    # initialize, setup games and controllers
    games = [
        Game({"random_state": 271828})
        for _ in range(2)
    ]
    controllers = [
        game.parallel_play()
        for game in games
    ]

    # go to first decision point
    [
        next(controller)
        for controller in controllers
    ]

    # modify cards in play
    for game in games:
        game.players[Identity.MIKE].cards[Zone.PLAY].append(
            Card(Name.MONK)
        )
    for _ in range(2):
        games[0].players[Identity.MIKE].cards[Zone.PLAY].append(
            Card(Name.GHOST)
        )
        games[1].players[Identity.MIKE].cards[Zone.PLAY].append(
            Card(Name.AKARI_TIMELESS_FIGHTER)
        )

    # get updated states
    for game in games:
        for player in game.players.values():
            player.unfreeze_state()
    states = [
        {
            identity: player.get_state()
            for identity, player in game.players.items()
        }
        for game in games
    ]
    for game in games:
        for player in game.players.values():
            player.freeze_state()

    # make strategies
    class ModifiedStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            offset = Zone.PLAY.value * len(Name)
            decision_matrix = X[:, offset: offset + NUM_CHOICES]
            decision_matrix[:, -1] = 0
            return decision_matrix
    strategies = {  # noqa
        identity: {
            "strategies": {
                action: ModifiedStrategy()
                for action in Action
            }
        }
        for identity in Identity
    }

    # run decision step
    Trainer._parallel_step(
        controllers=controllers,
        states=states,
        strategies=strategies,
    )

    # check result
    assert (
        games[0].players[Identity.MIKE].cards[Zone.DISCARD][0].name
        == Name.GHOST
    )
    assert (
        games[1].players[Identity.MIKE].cards[Zone.DISCARD][0].name
        == Name.AKARI_TIMELESS_FIGHTER
    )


def test__play_games():
    Trainer()._play_games()


def test__save_history__check_match():

    # initialize
    trainer = Trainer()
    game = Game({"random_state": 271828}).play()
    trainer._save_history([game])

    # check histories match
    for is_winner in [True, False]:

        # get current player
        player: Player = game.players[game.winner]
        if not is_winner:
            player = player.opponent
        identity = player.identity

        # check histories match
        for state in game.history.keys():
            for action in Action:
                if identity not in game.history[state][action]:
                    assert is_winner not in trainer.history[-1][state][action]
                    continue
                game_choices = game.history[state][action][identity]
                trainer_choices = trainer.history[-1][state][action][is_winner]
                assert len(game_choices) == len(trainer_choices)
                for g, t in zip(game_choices, trainer_choices):
                    assert g == t


def test__save_history__check_values():

    # initialize
    trainer = Trainer()
    game = Game()
    state0 = np.array([0., 2.]).tobytes()
    state1 = np.array([0., 1.]).tobytes()
    game.history = {
        state0: {
            action: {
                Identity.MIKE: [0],
                Identity.BRYAN: [1, 2, 3],
            }
            for action in Action
        },
        state1: {
            action: {
                Identity.MIKE: [5, 6, 7],
            }
            for action in Action
        },
    }
    game.winner = Identity.MIKE
    trainer._save_history([game])

    # check
    history = trainer.history[-1]
    for action in Action:
        assert history[state0][action][True] == [0]
        assert history[state0][action][False] == [1, 2, 3]
        assert history[state1][action][True] == [5, 6, 7]
        assert False not in history[state1][action]
