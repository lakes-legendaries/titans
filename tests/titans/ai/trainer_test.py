import numpy as np

from titans.ai import Game, Name, Network, Player, Trainer


def test__save_states():

    # initialize
    trainer = Trainer()
    game = Game({"random_state": 271828}).play()
    trainer._save_states(game)

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
