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

    This class trains up strategies for execellently-playing Titans of Eden. It
    works by executing a number of training epochs, wherein each epoch a bunch
    of games are played, and then data from those games is used to train up
    better and better strategies.

    During each epoch, the current strategy is played against the previous-best
    strategy to generate new training data.

    Early-stopping, weight restoration, and patience are all used to ensure the
    best strategy dominates. Callback is judged on the strategy that (1) plays
    the best against baseline, and (2) outperforms previous-best strategies.

    A temperature parameter is used and varied across games to make ensure that
    strategies don't get stuck in local minima.

    Parameters
    ----------
    baseline: bool, optional, default=False
        if True, only play against random-card-choosing strategies (instead of
        playing against latest-gen-minus-one)
    epochs: int, optional, default=10
        number of training epochs to execute
    games_per_epoch: int, optioanl, default=100
        number of games to play each epoch
    parallel: bool, optional, default=True
        play games in parallel, performing batch decision making at each
        decision point. This is much faster than making decisions
        one-at-a-time.
    retention: int, optional, default=3
        when training, keep game histories across this many epochs to train on.
        (i.e. training data from games more than `retention` epochs ago will be
        forgotten.)

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
    ):
        # save passed
        self._baseline = baseline
        self._epochs = epochs
        self._games_per_epoch = games_per_epoch
        self._parallel = parallel
        self._patience = patience
        self._retention = retention

        # initialize strategies
        self.strategies: dict[Action, Strategy] = {
            action: StandardStrategy() for action in Action
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
                            history.setdefault(state, {})
                            .setdefault(action, {})
                            .setdefault(is_winner, [])
                        ).extend(choices)

        # initialize Xy data
        Xy: dict[Action, tuple[list[np.ndarray], list[np.ndarray]]] = {
            action: ([], []) for action in Action
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
            action: (np.array(X), np.array(y)) for action, (X, y) in Xy.items()
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
        player_kwargs: dict[Identity, dict[str, dict[Action, Strategy]]],
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

        Completed games will be skipped.

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
                        player_kwargs.get(identity)
                        .get("strategies")
                        .get(action)
                        .predict(
                            np.vstack(
                                [
                                    state[identity]
                                    if state is not None
                                    else np.zeros(NUM_FEATURES)
                                    for state in states
                                ]
                            )
                        )
                    )
                )
                for action in Action
            }
            for identity in Identity
        }
        return [
            (
                controller.send(
                    {
                        identity: {
                            action: decision_matrices[identity][action][c]
                            for action in Action
                        }
                        for identity in Identity
                    }
                )
                if state is not None
                else None
            )
            for c, (controller, state) in enumerate(
                zip(
                    controllers,
                    states,
                )
            )
        ]

    def _play_games(
        self,
        /,
        *,
        save_history: bool = True,
        use_random: bool = False,
        vary_temperature: bool = False,
        vs_random: bool = False,
        vs_strategy: dict[Action, Strategy] | None = None,
    ) -> float:
        """Play games

        Player 0 will always use the strategies in self.strategies. Player 1
        will default to using the same strategies, but this can be overridden
        with `use_random` or `use_strategy`.

        This is an internal method that is called by `train`.

        Parameters
        ----------
        save_history: bool, optional, default=True
            save state history from these games
        use_random: bool, optional, default=False
            if True, then use random choices for both player 0 and player 1.
        vary_temperature: bool, optional, default=False
            vary temperature for each game, to make sure there's a good amount
            of noise (and thus varied decisions)
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
        random_strategy_dict = {action: RandomStrategy() for action in Action}
        player_kwargs: dict[Identity, dict[str, dict[Action, Strategy]]] = {
            Identity.MIKE: {
                "strategies": (
                    random_strategy_dict if use_random else self.strategies
                ),
            },
            Identity.BRYAN: {
                "strategies": (
                    random_strategy_dict
                    if vs_random
                    else vs_strategy
                    if vs_strategy is not None
                    else self.strategies
                ),
            },
        }

        # initialize games
        games = [
            self._init_game(
                {
                    **player_kwargs,
                    **(
                        {"temperature": temperature}
                        if vary_temperature
                        else {}
                    ),
                }
            )
            for temperature in np.linspace(0, 1, self._games_per_epoch)
        ]

        # play games sequentially
        if not self._parallel:
            games = [game.play() for game in games]

        # play games in parallel
        else:
            controllers = [game.parallel_play() for game in games]
            states = [next(controller) for controller in controllers]
            while any([state is not None for state in states]):
                states = self._parallel_step(
                    player_kwargs=player_kwargs,
                    controllers=controllers,
                    states=states,
                )

        # save game histories
        if save_history:
            self._save_history(games)

        # return win fraction
        return np.mean([game.winner == Identity.MIKE for game in games])

    def _save_history(self, games: list[Game], /) -> None:
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
                            history.setdefault(state, {})
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
        """Train network

        All parameters for training are set in `__init__`.
        """

        # initialize best strategy as empty standard strategy
        best_strategy = {action: StandardStrategy() for action in Action}

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
                if vs_baseline[idx] >= vs_baseline[best_idx] and (
                    self._baseline or vs_best[idx] > 0.50
                ):
                    best_idx = idx
            return best_idx

        # run training epochs
        for epoch in range(self._epochs):
            # train network
            if epoch:
                self._play_games(
                    vary_temperature=True,
                    vs_random=self._baseline,
                )
                Xy = self._get_Xy()
                for action in Action:
                    self.strategies[action].fit(*Xy[action])

            # test strategies
            vs_baseline.append(
                self._play_games(
                    save_history=False,
                    vs_random=True,
                )
            )
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
