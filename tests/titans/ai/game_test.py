import numpy as np

from titans.ai import (
    Action,
    Game,
    Identity,
    Name,
    NUM_CHOICES,
    Strategy,
)


def test___init__():

    # test basic construction
    game = Game()
    assert len(game.players) == 2
    assert len(game.players[Identity.MIKE].ritual_piles) == 4 * 4 * 4 + 12 + 8


def test__play_age():

    # set strategy to always choose to play Monk, and alway awaken Nikolai
    class ModifiedPlayStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.MONK.value] = 1
            return pred
    class ModifiedAwakenStrategy(Strategy):  # noqa
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.NIKOLAI_THE_CURSED.value] = 1
            return pred
    strategies = {
        Action.PLAY: ModifiedPlayStrategy(),
        Action.AWAKEN: ModifiedAwakenStrategy(),
    }

    # draw cards, play age
    game = Game({"strategies": strategies})
    for player in game.players.values():
        player.draw_cards(6)
    tuple(game._play_age())  # execute zero-lengthed generator

    # ensure correct cards played and awakened
    for player in game.players.values():
        assert player.play_zone[0].name == Name.MONK
        assert player.discard_zone[0].name == Name.NIKOLAI_THE_CURSED


def test__play_turn():

    # set strategies
    class ModifiedPlayStrategy(Strategy):
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.MONK.value] = 2
            pred[-1] = 1
            return pred
    class ModifiedAwakenStrategy(Strategy):  # noqa
        def predict(self, X: np.ndarray) -> np.ndarray:
            pred = np.zeros(NUM_CHOICES)
            pred[Name.NIKOLAI_THE_CURSED.value] = 1
            pred[Name.WINDS_HOWL.value] = 2
            pred[Name.FROSTBREATH.value] = 3
            return pred
    strategies = {
        Action.PLAY: ModifiedPlayStrategy(),
        Action.AWAKEN: ModifiedAwakenStrategy(),
    }

    # draw cards, play age
    game = Game(
        {"strategies": strategies, "random_state": 271828},
        {"random_state": 42},
    )
    tuple(game._play_turn())

    # ensure correct cards played and awakened
    assert all([
        card.name == Name.MONK
        for card in game.players[Identity.MIKE].play_zone
    ])
    discard_zone = game.players[Identity.MIKE].discard_zone
    assert discard_zone[0].name == Name.NIKOLAI_THE_CURSED
    assert discard_zone[1].name == Name.WINDS_HOWL
    assert discard_zone[2].name == Name.FROSTBREATH
    assert not all([
        c0.name == c1.name
        for c0, c1 in zip(*[
            game.players[identity].discard_zone
            for identity in Identity
        ])
    ])

    # play a few more turns, check temple count
    for _ in range(10):
        tuple(game._play_turn())
    assert (
        (game.players[Identity.MIKE].temples < 3)
        or (game.players[Identity.BRYAN].temples < 3)
    )

    # check history
    choices = list(game.history[Identity.MIKE][Action.AWAKEN].values())
    assert len(choices[0]) == 1
    assert choices[0][0] == Name.NIKOLAI_THE_CURSED.value
    assert len(choices[1]) == 1
    assert choices[1][0] == Name.WINDS_HOWL.value
    assert len(choices[2]) == 1
    assert choices[2][0] == Name.FROSTBREATH.value


def test_parallel_play():

    # initialize game
    game = Game()
    controller = game.parallel_play()

    # check initial states
    states = next(controller)
    assert len(states) == 2
    assert states[0][-1] == 6
    assert states[0][-2] == 6

    # send choice to play monk
    decision_matrix = {action: np.zeros(NUM_CHOICES) for action in Action}
    decision_matrix[Action.PLAY][Name.MONK.value] = 1
    controller.send([decision_matrix] * 2)

    # check correct card played
    for player in game.players.values():
        assert player.play_zone[0].name == Name.MONK


def test_play():
    game = Game().play()
    assert game.players[game.winner].temples > 0
    assert game.players[game.winner].opponent.temples == 0
