from titans.ai import Game


def test___init__():
    game = Game()
    assert len(game.players) == 2
    assert len(game.players[0].ritual_piles) == 4 * 4 * 4 + 12 + 8
