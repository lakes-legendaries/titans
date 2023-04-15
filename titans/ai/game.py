"""Game Module"""

from __future__ import annotations

from typing import Any, Generator

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Action, Name, Identity
from titans.ai.player import Player


class Game:
    """Game Class

    Parameters
    ----------
    player_kwargs: dict[str, Any] | dict[Identity, dict[str, Any]]
        These dictionaries are unpacked as kwargs to initialize the players. If
        you provide a dictionary of strings, then the provided values will be
        unpacked to initialize both players. If you provide a dictionary of
        identities mapping to a dictionary of strings, then each player will be
        initialized with the corresponding kwargs provided for that player.
    turn_limit: int, optional, default=1000
        max number of turns before a draw is declared

    Attributes
    ----------
    cards: list[Card]
        cards in the game
    players: list[Players]
        players playing the game
    history: dict[Identity, dict[Action, dict[bytes, list[int]]]]
        here, the history of each player's state, and the choices they made
        given that state, is recorded. This variable contains three nested
        dictionaries:

        1. The top-level dictionary is indexed by each player
        2. The mid-level dictionary is indexed by each action
        3. The bottom-level dictionary maps from game state to the choices made

    winner: Identity | None
        winner of game
    """
    def __init__(
        self,
        player_kwargs: dict[str, Any] | dict[Identity, dict[str, Any]] = None,
        /,
        *,
        turn_limit: int = 1000,
    ):
        # save parameters
        self._turn_limit = turn_limit

        # construct cards
        self.cards: list[Card] = []
        for name in Name:
            count = 4
            match name:
                case Name.MONK:
                    count = 16
                case Name.WIZARD | Name.TRAVELER:
                    count = 8
                case Name.GHOST:
                    count = 12
            self.cards.extend([Card(name) for _ in range(count)])

        # construct players
        self.players: dict[Identity, Player] = {
            identity: Player(
                identity,
                cards=self.cards,
                **(
                    {}
                    if player_kwargs is None
                    else player_kwargs
                    if identity not in player_kwargs
                    else player_kwargs[identity]
                ),
            )
            for identity in Identity
        }
        self.players[Identity.MIKE].handshake(self.players[Identity.BRYAN])

        # initialize history tracking
        self.history: dict[Identity, dict[Action, dict[bytes, list[int]]]] = {
            identity: {
                action: {}
                for action in Action
            }
            for identity in Identity
        }
        self.winner: Identity | None = None

    def _play_age(
        self,
        use_generators: bool = False,
    ) -> Generator[
        dict[Identity, np.array],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Execute an age

        This function is modularly designed to either yield a generator that
        you can interact with (if you set `use_generators=True`), or to return
        a zero-length generator so you can execute the whole age with
        `tuple(Game._play_age())`. It's set up this way so that you have the
        option to run many games in parallel with one another, syncing
        decisions across games (which allows for much faster decision matrix
        computation).

        If you are using this as a generator, the function will yield at each
        decision point, and let you send in the decision each player is to
        make.

        Parameters
        ----------
        use_generators: bool, optional, default=False
            If True, then yield an interactive generator; if False, yield a
            zero-length generator that executes the whole method.

        Returns
        -------
        Generator
            This generator yields each player's state, and sends precomputed
            decision matrices for each player. If `use_generators=False`, then
            this will be a zero-length generator.
        """

        # freeze states
        for player in self.players.values():
            player.freeze_state()

        # yield player states, make decisions outside of this game
        if use_generators:
            decision_matrices = \
                yield {
                    identity: player._frozen_state
                    for identity, player in self.players.items()
                }
            for identity, matrix in decision_matrices.items():
                self.players[identity]._decision_matrices = matrix

        # play and awaken cards, saving states
        for player in self.players.values():
            frozen_state = player._frozen_state.tobytes()
            for method, action in [
                (Player.play_cards, Action.PLAY),
                (Player.awaken_card, Action.AWAKEN),
            ]:
                _, choice = method(player)
                state_dict = (
                    self.history[player.identity][action]
                    .setdefault(frozen_state, [])
                )
                if type(choice) is list:
                    state_dict.extend(choice)
                else:
                    state_dict.append(choice)

        # unfreeze states
        for player in self.players.values():
            player.unfreeze_state()

        # void out decision matrices
        if use_generators:
            for player in self.players.values():
                player._decision_matrices = None

        # stop iteration
        yield from []

    def _play_turn(
        self,
        use_generators: bool = False,
    ) -> Generator[
        dict[Identity, np.array],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Execute a complete turn"""

        # shuffle step (we do this first)
        for player in self.players.values():
            player.shuffle_cards()
            player.draw_cards(6)

        # play ages
        for _ in range(3):
            yield from self._play_age(use_generators=use_generators)

        # battle
        self.players[Identity.MIKE].battle_opponent()

    def _play(
        self,
        use_generators: bool = False,
    ) -> Generator[
        dict[Identity, np.array],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Play game"""

        # play game
        for _ in range(self._turn_limit):
            yield from self._play_turn(use_generators=use_generators)
            for player in self.players.values():
                if player.temples <= 0:
                    self.winner = player.opponent.identity
                    return self

        # no winner after N turns -- tie game
        self.winner = None
        return self

    def parallel_play(self) -> Generator[
        dict[Identity, np.array],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ] | None:
        """Play game, returning a generator that pauses at each decision point
        """
        yield from self._play(use_generators=True)
        yield None

    def play(self) -> Game:
        """Play game

        Returns
        -------
        Game
            calling instance
        """
        tuple(self._play(use_generators=False))
        return self
