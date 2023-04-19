"""Trainer module"""

from collections import deque
from typing import Any, Generator

import numpy as np

from titans.ai.constants import NUM_CHOICES, NUM_FEATURES
from titans.ai.enum import Ability, Action, Identity
from titans.ai.game import Game
from titans.ai.strategy import RandomStrategy, StandardStrategy, Strategy


class Trainer:
    """Standard trainer

    Parameters
    ----------
    retention: int, optional, default=10
        number of game histories to keep

    Attributes
    ----------
    history: deque[dict[bytes, dict[Action, dict[bool, list[int]]]]]
        here, the history of winning and losing players is recorded. This
        variable maps every state to every choice made given that state,
        recording whether the player that made that choice ultimately won or
        lost the game.

        This is saved as a circular buffer (deque), so that the Trainer can
        retain the histories from the most recent N games.
    """
    def __init__(
        self,
        *,
        retention: int = 3,
    ):
        # initialize strategies
        self.strategies: dict[Action, Strategy] = {
            action: StandardStrategy()
            for action in Action
        }

        # initialize history
        self.history: (
            deque[dict[bytes, dict[Action, dict[bool, list[int]]]]]
        ) = deque(maxlen=retention)

    def _init_game(
        self,
        player_kwargs: dict[str, Any] | dict[Identity, dict[str, Any]],
    ) -> Game:
        """Initialize game object

        Parameters
        ----------
        *args: Any
            passed to `Game()` on `__init__()`

        Returns
        -------
        Game
            game, already played
        """
        return Game(player_kwargs)

    @classmethod
    def _parallel_step(
        cls,
        strategies: dict[Identity, dict[str, dict[Action, Strategy]]],
        controllers: list[
            Generator[
                dict[Identity, np.array] | None,
                dict[Identity, dict[Action, np.ndarray]],
                None,
            ]
        ],
        states: list[dict[Identity, np.ndarray] | None],
    ) -> list[dict[Identity, np.ndarray] | None]:
        """Run one decision point of a game played in parallel

        Completed games will be passed over.

        This function is broken out for easy testing of this
        somewhat-complicated method.

        Parameters
        ----------
        strategies: dict[Identity, dict[str, dict[Action, Strategy]]]
            strategies passed to games being played. This method expects that
            all games played in parallel have the same strategies (although
            players within a game can have different strategies).
        controllers: list[Generator]
            generators created from Game.parallel_play()
        states: list[dict[Identity, np.ndarray] | None]
            game states yielded from controllers

        Returns
        -------
        list[dict[Identity, np.ndarray] | None]
            updated states, after running one decision step in parallel.
        """
        decision_matrices = {
            identity: {
                action: (
                    (
                        strategies
                        .get(identity)
                        .get("strategies")
                        .get(action)
                        .predict(np.vstack([
                            state[identity]
                            if state is not None
                            else np.zeros(NUM_FEATURES)
                            for state in states
                        ]))
                    )
                )
                for action in Action
            }
            for identity in Identity
        }
        return [
            (
                controller.send({
                    identity: {
                        action: decision_matrices[identity][action][c]
                        for action in Action
                    }
                    for identity in Identity
                })
                if state is not None
                else None
            )
            for c, (controller, state) in enumerate(zip(
                controllers,
                states,
            ))
        ]

    def _save_history(self, games: list[Game]):
        """Extract and save histories from games

        Parameters
        ----------
        games: list[Game]
            most recently-played game
        """

        # initialize overall history dictionary
        history: dict[bytes, dict[Action, dict[bool, list[int]]]] = {}

        # combine history from all games. index by winner (instead of identity)
        for game in games:
            for state, state_dict in game.history.items():
                for action, action_dict in state_dict.items():
                    for identity, choices in action_dict.items():
                        is_winner = identity == game.winner
                        (
                            history
                            .setdefault(state, {})
                            .setdefault(action, {})
                            .setdefault(is_winner, [])
                        ).extend(choices)

        # save history
        self.history.append(history)

    def get_Xy(self) -> dict[Action, tuple[np.ndarray, np.ndarray]]:
        """Transform self.history -> (X, y) data

        Expect lots of np.NaN values in y! Additionally, y data will be brought
        in by the std error (see err_bound_low and err_bound_high in
        __init__()).

        Returns
        -------
        dict[Action, tuple[np.ndarray, np.ndarray]]
            dictionary mapping from action -> (X, y) data
        """

        # enable divide-by-zero without warning
        np.seterr(divide="ignore", invalid="ignore")

        # combine history from all recent games
        history: dict[bytes, dict[Action, dict[bool, list[int]]]] = {}
        for epoch in self.history:
            for state, state_dict in epoch.items():
                for action, action_dict in state_dict.items():
                    for is_winner, choices in action_dict.items():
                        (
                            history
                            .setdefault(state, {})
                            .setdefault(action, {})
                            .setdefault(is_winner, [])
                        ).extend(choices)

        # initialize Xy data
        Xy: dict[Action, tuple[list[np.ndarray], list[np.ndarray]]] = {
            action: ([], [])
            for action in Action
        }

        # process each action and each state
        for state, state_dict in history.items():
            for action, action_dict in state_dict.items():

                # translate wins and losses from list[int] -> np.ndarray
                win_loss_count: dict[bool, np.ndarray] = {
                    is_winner: np.zeros(NUM_CHOICES)
                    for is_winner in [True, False]
                }
                for is_winner, choices in action_dict.items():
                    for choice in choices:
                        win_loss_count[is_winner][choice] += 1

                # get means and std err
                counts = win_loss_count[True] + win_loss_count[False]
                means = win_loss_count[True] / counts

                # save results
                Xy[action][0].append(np.frombuffer(state))
                Xy[action][1].append(means)

        # return, converting from list[1d np.ndarrays] to 2d np.ndarrays
        return {
            action: (np.array(X), np.array(y))
            for action, (X, y) in Xy.items()
        }

    def play(
        self,
        /,
        *,
        num_games: int = 100,
        parallel: bool = False,
        save_history: bool = True,
        use_random: bool = False,
        vs_random: bool = False,
        vs_strategy: dict[Action, Strategy] | None = None,
    ) -> float:
        """Play a game with random strategies

        Player 0 will always use the strategies in self.strategies. Player 1
        will default to using the same strategies, but this can be overridden
        with `use_random` or `use_strategy`.

        Parameters
        ----------
        num_games: int, optional, default=1000
            number of games to play each session
        parallel: bool, optional, default=False
            play games in parallel (much faster)
        save_history: bool, optional, default=True
            save state history from these games
        use_random: bool, optional, default=False
            if True, then use random choices for both player 0 and player 1.
        vs_random: bool, optional, default=False
            if True, instead of using `self.strategies` as player 1's strategy,
            use random choices.
        vs_strategy: dict[Action, Strategy] | None, optional, default=None
            if provided, instead of using `self.strategies` as player 1's
            strategy, use the provided strategy here.

        Returns
        -------
        float
            fraction of games won by player 0
        """  # noqa

        # check args
        if vs_random and vs_strategy is not None:
            raise ValueError(
                "Conflicting strategies provided."
                " You can set vs_random or vs_strategy,"
                " but not both."
            )

        # get strategies
        random_strategy_dict = {
            action: RandomStrategy()
            for action in Action
        }
        strategies: dict[Identity, dict[str, dict[Action, Strategy]]] = {
            Identity.MIKE: {
                "strategies": (
                    random_strategy_dict
                    if use_random
                    else self.strategies
                )
            },
            Identity.BRYAN: {
                "strategies": (
                    random_strategy_dict
                    if vs_random
                    else vs_strategy
                    if vs_strategy is not None
                    else self.strategies
                ),
            }
        }

        # play games sequentially
        if not parallel:
            games = [
                self._init_game(strategies).play()
                for _ in range(num_games)
            ]

        # play games in parallel
        else:
            games = [self._init_game(strategies) for _ in range(num_games)]
            controllers = [game.parallel_play() for game in games]
            states = [next(controller) for controller in controllers]
            while any([state is not None for state in states]):
                states = self._parallel_step(
                    strategies=strategies,
                    controllers=controllers,
                    states=states,
                )

        # save game histories
        if save_history:
            self._save_history(games)

        # return win fraction
        return np.mean([game.winner == Identity.MIKE for game in games])

    def train(self):
        """Train network"""
        Xy = self.get_Xy()
        for action in Action:
            self.strategies[action].fit(*Xy[action])


class POCTrainer(Trainer):
    """Simple proof-of-concept trainer that demonstrates learning

    This trainer removes all non-Energy abilities from the cards before playing
    """
    def _init_game(self) -> Game:
        game = Game()
        for card in game.cards:
            card.abilities = {
                ability: value
                for ability, value in card.abilities.items()
                if ability == Ability.ENERGY
            }
        return game
