"""Trainer module"""

import numpy as np

from titans.ai.enum import Ability, Name, Network
from titans.ai.game import Game


class Trainer:
    """Standard trainer

    Parameters
    ----------
    clip_exclusion_zone: float, optional, default=0.2
        when computing the win percentage for any given choice, we bring it in
        towards 0.5 by the standard error. (E.g. if, for a given state, we make
        a choice 4 times, and we win all 4 times, then the y-label for that
        instance will be 1.0 - 1/sqrt(4)). Values less than 0.5 are raised, and
        values greater than 0.5 are lowered. However, we'll only bring in by so
        much (because we don't want, say, something with a win percentage of
        0.55 to be brought to the other side of 0.50).

        The clip_exclusion_zone is the window surrounding 0.5 that values won't
        be brought beyond. E.g. if clip_exclusion_zone=0.2, then values won't
        be raised above 0.4 and lowered beneath 0.6.

        If you set this to None, then no clipping will occur.

    Attributes
    ----------
    history: dict[bool, dict[Network, dict[bytes, np.ndarray]]]
        here, the history of winning and losing players is recorded. This
        variable maps every global state to every choice made given that state,
        recording whether the player that made that choice ultimately won or
        lost the game.

        This variable contains three nested dictionaries:

        2. The top-level dictionary shows whether the state came from the
           winning or losing player. (Please note that states can exist in both
           the winning and losing dictionaries, and you have to compare these
           to get the win percentage for any given move.)
        2. The mid-level dictionary is indexed by each strategy (e.g. awaken,
           play)
        3. The bottom-level dictionary maps from game state to the number of
           times each choice was made

        There are two differences between Trainer.history and Game.history:

        1. Game.history lists the choices of each player, while Trainer.history
           combines across winning and losing players.
        2. While the bottom-level dictionary of Game.history contains a
           list[int] that shows each choice that was made, the equivalent
           Trainer.history dictionary contains an np.ndarray that shows the
           number of times each choice was made.
    """
    def __init__(
        self,
        *,
        clip_exclusion_zone: float = 0.2,
    ):
        # save passed
        self._clip_exclusion_zone = clip_exclusion_zone

        # initialize history dictionary
        self.history: dict[bool, dict[Network, dict[bytes, np.ndarray]]] = {
            is_winner: {
                network: {}
                for network in Network
            }
            for is_winner in [True, False]
        }

    def _play_game(self) -> Game:
        """Play game

        Returns
        -------
        Game
            game, already played
        """
        return Game.play()

    def _save_history(self, game: Game):
        """Extract and save history from the most recently-played game

        Parameters
        ----------
        game: Game
            most recently-played game
        """
        default_zeros = np.zeros(len(Name) + 1)
        for identity, player_history in game.history.items():
            is_winner = identity == game.winner
            for network, network_history in player_history.items():
                for state, choices in network_history.items():

                    # add default zeros to dictionary
                    (
                        self
                        .history[is_winner][network]
                        .setdefault(state, default_zeros.copy())
                    )

                    # increment choices
                    for choice in choices:
                        self.history[is_winner][network][state][choice] += 1

    def get_Xy(self) -> dict[Network, tuple[np.ndarray, np.ndarray]]:
        """Transform self.history -> (X, y) data

        Expect lots of np.NaN values in y! Additionally, y data will be brought
        in by the std error (see err_bound_low and err_bound_high in
        __init__()).

        Returns
        -------
        dict[Network, tuple[np.ndarray, np.ndarray]]
            dictionary mapping from strategy (network) -> (X, y) data
        """

        # enable divide-by-zero without warning
        np.seterr(divide="ignore", invalid="ignore")

        # default counts for value-missing
        default_zeros = np.ndarray(len(Name) + 1)

        # do for each strategy
        Xy = {}
        for network in Network:

            # get list of all states
            all_states = np.unique(np.concatenate([
                list(self.history[is_winner][network].keys())
                for is_winner in [True, False]
            ]))

            # initialize X, y data
            X = []
            y = []

            # process each state
            for state in all_states:

                # get wins and counts
                wins = self.history[True][network].get(state, default_zeros)
                losses = self.history[False][network].get(state, default_zeros)
                counts = wins + losses

                # get means and std err
                means = wins / counts
                std_err = 1 / counts

                # clip scores
                margin = self._clip_exclusion_zone / 2
                scores = means.copy()
                for raise_values in [False, True]:
                    bound = (
                        (np.subtract if raise_values else np.add)
                        (0.5, margin)
                    )
                    mod_idx = (
                        (np.less if raise_values else np.greater)
                        (means, bound)
                    )
                    scores[mod_idx] = (
                        (np.minimum if raise_values else np.maximum)
                        (
                            (
                                (np.add if raise_values else np.subtract)
                                (scores[mod_idx], std_err[mod_idx])
                            ),
                            bound,
                        )
                    )

                # save results
                X.append(np.frombuffer(state))
                y.append(scores)

            # save as arrays
            Xy[network] = (np.array(X), np.array(y))

        # return
        return Xy

    def play_random(self, num_games: int = 100):
        """Play a game with random strategies

        Parameters
        ----------
        num_games: int
            number of games to play each session
        """

        # play games, save states
        for _ in range(num_games):
            game = self._play_game()
            self._save_history(game)


class POCTrainer(Trainer):
    """Simple proof-of-concept trainer that demonstrates learning

    This trainer removes all non-Energy abilities from the cards before playing
    """
    def _play_game(self) -> Game:
        game = Game()
        for card in game.cards:
            card.abilities = {
                ability: value
                for ability, value in card.abilities.items()
                if ability == Ability.ENERGY
            }
        return game.play()
