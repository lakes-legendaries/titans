"""Trainer module"""

import numpy as np

from titans.ai.enum import Ability, Name, Network
from titans.ai.game import Game


class Trainer:
    """Standard trainer

    Parameters
    ----------
    err_bound_low: float, optional, default=0.4
        when computing the win percentage for any given choice, we bring it in
        towards 0.5 by the standard error. (E.g. if, for a given state, we make
        a choice 4 times, and we win all 4 times, then the y-label for that
        instance will be 1.0 - 1/sqrt(4)). Values less than 0.5 are raised, and
        values greater than 0.5 are lowered. However, we'll only lower so much
        (because we don't want, say, something with a win percentage of 0.55 to
        be brought to the other side of 0.50). This value is the upper bound to
        which we'll raise any observable to when applying the standard error.
    err_bound_high: float, optional, default=0.6
        same as err_bound_low, but this is the lower bound we'll lower any
        value to.

    Attributes
    ----------
    states: dict[Network, dict[bool, dict[bytes, np.ndarray]]]
        This dictionary lets you see each choice that was made in each game,
        and whether it led to winning or losing the game. It contains three
        nested dictionaries:

        1. The top-level dictionary is indexed by the network (e.g. play,
           awaken).
        2. The mid-level dictionary shows whether the state came from the
           winning or losing player. (Please note that states can exist in both
           the winning and losing dictionaries, and you have to compare these
           to get the win percentage for any given move.)
        3. The bottom-level dictionary maps from the state to the number of
           times each choice was made.
    """
    def __init__(
        self,
        *,
        err_bound_low: float = 0.4,
        err_bound_high: float = 0.6,
    ):
        # save passed
        self._err_bound_low = err_bound_low
        self._err_bound_high = err_bound_high

        # initialize states dictionary
        self.states: dict[Network, dict[bool, dict[bytes, np.ndarray]]] = {
            network: {
                is_winner: {}
                for is_winner in [True, False]
            }
            for network in Network
        }

    def _get_Xy(self) -> dict[Network, tuple[np.ndarray, np.ndarray]]:
        """Transform self.states -> (X, y) data

        Expect lots of np.NaN values in y! Additionally, y data will be brought
        in by the std error.

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
        for network, network_states in self.states.items():

            # get list of states
            states = np.concatenate([
                list(network_states[is_winner].keys())
                for is_winner in [True, False]
            ])

            # initialize X, y data
            X = []
            y = []

            # process each state
            for state in states:

                # get wins and counts
                wins = network_states[True].get(state, default_zeros)
                counts = wins + network_states[False].get(state, default_zeros)

                # get means and std err
                means = wins / counts
                std_err = 1 / counts

                # clip scores
                bring_up = means < self._err_bound_low
                bring_down = means > self._err_bound_high
                scores = means.copy()
                scores[bring_up] = np.minimum(
                    scores[bring_up] + std_err[bring_up],
                    self._err_bound_low,
                )
                scores[bring_down] = np.maximum(
                    scores[bring_down] - std_err[bring_down],
                    self._err_bound_high,
                )

                # save results
                X.append(np.frombuffer(state))
                y.append(scores)

            # save as arrays
            Xy[network] = (np.array(X), np.array(y))

        # return
        return Xy

    def _play_game(self) -> Game:
        """Play game

        Returns
        -------
        Game
            game, already played
        """
        return Game.play()

    def _save_states(self, game: Game):
        default_zeros = np.ndarray(len(Name) + 1)
        for identity, player_states in game.states.items():
            is_winner = identity == game.winner
            for network, network_states in player_states.items():
                for state, choices in network_states.items():

                    # add key to overall dict
                    (
                        self
                        .states[network][is_winner]
                        .setdefault(state, default_zeros.copy())
                    )

                    # increment choices
                    for choice in choices:
                        self.states[network][is_winner][state][choice] += 1

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
            self._save_states(game)

        # # create Xy data
        Xy = self._get_Xy()


class AwakenTrainer(Trainer):
    """This trainer uses ability-less cards for a simple POC"""
    def _play_game(self) -> Game:
        game = Game()
        for card in game.cards:
            card.abilities = {
                ability: value
                for ability, value in card.abilities.items()
                if ability == Ability.ENERGY
            }
        return game.play()


AwakenTrainer().play_random(100)
