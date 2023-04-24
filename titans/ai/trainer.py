"""Trainer module"""

from collections import deque
from copy import deepcopy
from typing import Any, Generator, Self

import numpy as np

from titans.ai.constants import NUM_CHOICES, NUM_FEATURES
from titans.ai.enum import Ability, Action, Identity, Name
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
    strategies: dict[Action, Strategy]
        Strategies trained by this class
    """
    def __init__(
        self,
        /,
        *,
        baseline: bool = False,
        epochs: int = 10,
        games_per_epoch: int = 100,
        parallel: bool = True,
        patience: int = 3,
        retention: int = 3,
        verbose: bool = False,
    ):
        # save passed
        self._baseline = baseline
        self._epochs = epochs
        self._games_per_epoch = games_per_epoch
        self._parallel = parallel
        self._patience = patience
        self._retention = retention
        self._verbose = verbose

        # initialize strategies
        self.strategies: dict[Action, Strategy] = {
            action: StandardStrategy()
            for action in Action
        }

        # initialize history
        self.history: (
            deque[dict[bytes, dict[Action, dict[bool, list[int]]]]]
        ) = deque(maxlen=retention)

    def _get_Xy(self, /) -> dict[Action, tuple[np.ndarray, np.ndarray]]:
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

    def _init_game(
        self,
        player_kwargs: dict[str, Any] | dict[Identity, dict[str, Any]],
        /,
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
        /,
        *,
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

    def _play_games(
        self,
        /,
        *,
        save_history: bool = True,
        temperature: float | None = None,
        use_random: bool = False,
        vs_random: bool = False,
        vs_strategy: dict[Action, Strategy] | None = None,
    ) -> float:
        """Play games

        Player 0 will always use the strategies in self.strategies. Player 1
        will default to using the same strategies, but this can be overridden
        with `use_random` or `use_strategy`.

        Parameters
        ----------
        save_history: bool, optional, default=True
            save state history from these games
        temperature: float | None, optional, default=None
            stochastic noise to add to players' decisions
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
                ),
                "temperature": temperature,
            },
            Identity.BRYAN: {
                "strategies": (
                    random_strategy_dict
                    if vs_random
                    else vs_strategy
                    if vs_strategy is not None
                    else self.strategies
                ),
                "temperature": temperature,
            },
        }

        # play games sequentially
        if not self._parallel:
            games = [
                self._init_game(strategies).play()
                for _ in range(self._games_per_epoch)
            ]

        # play games in parallel
        else:
            games = [
                self._init_game(strategies)
                for _ in range(self._games_per_epoch)
            ]
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

    def _save_history(self, games: list[Game], /):
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

    def discover(self, /) -> np.ndarray:
        """Discover best cards

        This function uses this trainer's strategy to weight each card
        according to its relative value.

        Returns
        -------
        np.ndarray of shape(len(Name),) of dtype float
            weights (relative value) of each card
        """
        weights = np.zeros(len(Name))
        for name in Name:
            vs_strategy = deepcopy(self.strategies)
            for strategy in vs_strategy.values():
                strategy.restricted = [name.value]
            weights[name.value] = self._play_games(
                save_history=False,
                vs_strategy=vs_strategy,
            )
        return weights

    def train(self, /) -> Self:
        """Train network"""

        # initialize best strategy as empty standard strategy
        best_strategy = {
            action: StandardStrategy()
            for action in Action
        }

        # initialize metrics (win trackers)
        vs_baseline = []
        vs_best = []
        self.metrics = {
            "vs_baseline": vs_baseline,
            "vs_best": vs_best,
        }

        # function to check for best
        def get_best_idx() -> int:
            """Get best strategy

            The best strategy is the one that both has (1) the highest win
            fraction against random plays, and simultaneously (2) outperforms
            the previous top-performing strategy.

            Technicaly, this isn't *really* finding the best strategy; it's
            more of a patience checker, as it finds the most recent strategy
            that satisifes all criterion, and gives ties to the oldest. You can
            then compare the best idx to the current idx, and see if the
            patience counter has been exceeded.

            Returns
            -------
            int
                round number of top-performing strategy
            """
            best_idx = 0
            for idx in range(1, len(vs_baseline)):
                if (
                    vs_baseline[idx] >= vs_baseline[best_idx]
                    and (
                        self._baseline
                        or vs_best[idx] > 0.50
                    )
                ):
                    best_idx = idx
            return best_idx

        # run training epochs
        for epoch in range(self._epochs):

            # train network
            if epoch:
                self._play_games(
                    temperature=0.1,
                    vs_random=self._baseline,
                )
                Xy = self._get_Xy()
                for action in Action:
                    self.strategies[action].fit(*Xy[action])

            # test strategies
            vs_baseline.append(self._play_games(
                save_history=False,
                vs_random=True,
            ))
            if not self._baseline:
                vs_best.append(
                    self._play_games(
                        save_history=False,
                        vs_strategy=best_strategy,
                    )
                )

            # if best scores yet, export strategy as best
            if (best_idx := get_best_idx()) == len(vs_baseline) - 1:
                best_strategy = deepcopy(self.strategies)

            # if not best, restore best strategy, check for early stopping
            else:
                self.strategies = deepcopy(best_strategy)
                if best_idx < len(vs_baseline) - self._patience:
                    break

        # return
        return self


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
