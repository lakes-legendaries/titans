"""Game Module"""

from __future__ import annotations

from typing import Any, Generator

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Action, Name, Identity, Zone
from titans.ai.player import Player


class Game:
    """Game Class

    This class contains all the logic to play a full game.

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
    history: dict[bytes, dict[Action, dict[Identity, list[int]]]]
        here, the history of each player's state, and the choices they made
        given that state, are recorded. This variable contains three nested
        dictionaries:

        1. The top-level dictionary is indexed by each game state at which a
           decision was made (converted from np.ndarrary -> bytes, for
           hashability)
        2. The mid-level dictionary is indexed by each action made at that
           decision point
        3. The bottom-level dictionary is indexed by the player that made that
           action, and maps to the choice(s) that player made at that decision
           point

    players: list[Players]
        players playing the game
    transcript: str
        human-readable transcript (log) of the game
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
        self.history: dict[bytes, dict[Action, dict[Identity, list[int]]]] = {}
        self.winner: Identity | None = None
        self.transcript = ""

    def _play_age(
        self,
        use_generators: bool = False,
    ) -> Generator[
        dict[Identity, np.array],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Execute an age

        Parameters
        ----------
        use_generators: bool, optional, default=False
            If True, then yield an interactive generator; if False, yield a
            zero-length generator that executes the whole method.

        Returns
        -------
        Generator
            A generator that yields at each decision point (if
            `use_generators`; otherwise, generator is zero-length). See `_play`
            for more details.
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
        for identity, player in self.players.items():
            frozen_state = player._frozen_state.tobytes()
            for method, action in [
                (Player.play_cards, Action.PLAY),
                (Player.awaken_card, Action.AWAKEN),
            ]:
                # do action, get consistent formatting
                cards, choices = method(player)
                if type(choices) is not list:
                    choices = [choices]
                    cards = [cards]

                # update state dictionary
                state_dict = (
                    self.history
                    .setdefault(frozen_state, {})
                    .setdefault(action, {})
                    .setdefault(identity, [])
                )
                state_dict.extend(choices)

                # update transcript
                self.transcript += (
                    "        "
                    + f"{identity.name.title():5s}"
                    + f" {action.name.lower():6s}'d "
                    + ", ".join([
                        (
                            card.name.name
                            if card is not None
                            else "None"
                        ) + (
                            ""
                            if choice < len(Name)
                            else " (default)"
                        )
                        for choice, card in zip(choices, cards)
                    ]) + "\n"
                )

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
        dict[Identity, np.ndarray],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Execute a complete turn

        Parameters
        ----------
        use_generators: bool, optional, default=False
            if True, yield at each decision point. Otherwise, play end-to-end
            (i.e. yielding a zero-length generator).

        Returns
        -------
        Generator
            A generator that yields at each decision point (if
            `use_generators`; otherwise, generator is zero-length). See `_play`
            for more details.
        """

        # shuffle step (we do this first)
        for player in self.players.values():
            player.shuffle_cards()
            player.draw_cards(6)

        # play ages
        for age_num in range(3):
            self.transcript += f"    Age {age_num + 1}\n"
            yield from self._play_age(use_generators=use_generators)

        # battle
        self.transcript += "    Battle\n"
        for identity, player in self.players.items():
            self.transcript += (
                f"        {identity.name.title():5s}'s power:"
                f" {player.get_power()}\n"
            )
        winner = self.players[Identity.MIKE].battle_opponent()
        if winner is not None:
            self.transcript += f"        Winner: {winner.name}\n"
            for identity, player in self.players.items():
                self.transcript += (
                    f"        {identity.name.title()}'s temples:"
                    f" {player.temples}\n"
                )

        # overview
        self.transcript += "    Overview\n"
        for zone in Zone:
            self.transcript += f"{' ':8s}{zone.name.title()}\n"
            for identity, player in self.players.items():
                self.transcript += f"{' ':12s}{identity.name.title()}\n"
                self.transcript += f"{' ':16s}{player.cards[zone]}\n"

    def _play(
        self,
        use_generators: bool = False,
    ) -> Generator[
        dict[Identity, np.ndarray],
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Play game

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

        Returns
        -------
        Generator
            A generator that yields at each decision point (if
            `use_generators`; otherwise, generator is zero-length). The only
            difference between the return here and the return in
            `parallel_play` is that this function will never yield `None`; that
            is added by `parallel_play` once the game completes. See
            `parallel_play` for more information.
        """

        # play game
        for turn_num in range(self._turn_limit):
            self.transcript += f"Turn {turn_num + 1}\n"
            yield from self._play_turn(use_generators=use_generators)
            for player in self.players.values():
                if player.temples <= 0:
                    self.winner = player.opponent.identity
                    return self

        # no winner after N turns -- tie game
        self.winner = None
        return self

    def parallel_play(self) -> Generator[
        dict[Identity, np.ndarray] | None,
        dict[Identity, dict[Action, np.ndarray]],
        None,
    ]:
        """Play game, returning a generator that pauses at each decision point

        This lets an operator simultaneously make decisions across many
        different games. This is important because the slowest part of
        processing is using the ANNs to make decisions. So, using this function
        lets operators speed up game simulations considerably.

        Returns
        -------
        Generator
            This generator will play the game to each decision point, and will
            then expect you to send in the `decision_matrices` that rank each
            possible choice.

            Yields: dict[Identity, np.ndarray] | None
                dictionary mapping from each player to their current state. If
                None is yielded, the game is over.
            Send: dict[Identity, dict[Action, np.ndarray]]
                dictionary mapping from each player to (
                    dictionary mapping from each action to the choices that
                    player should make for that action, wherein each possible
                    choice is ranked by its relative value. (highest value =
                    perform that decision)
                )
            Returns: None
        """
        yield from self._play(use_generators=True)
        yield None

    def play(self) -> Game:
        """Play game end-to-end

        Returns
        -------
        Game
            calling instance
        """
        tuple(self._play(use_generators=False))
        return self
