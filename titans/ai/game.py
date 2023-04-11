"""Game Module"""

from __future__ import annotations

from typing import Any

from titans.ai.card import Card
from titans.ai.enum import Name, Network, Identity
from titans.ai.player import Player


class Game:
    """Game Class

    Parameters
    ----------
    *args: dict[str, Any] | list[dict[str, Any]]
        These dictionaries are unpacked to initialize the players. If one arg
        is provided, then this is used to initialize both players. If two are
        provided, then one is used for each player.
    turn_limit: int, optional, default=1000
        max number of turns before a draw is declared

    Attributes
    ----------
    cards: list[Card]
        cards in the game
    players: list[Players]
        players playing the game
    states: list[list[dict[bytes, list[int | list[int]]]]]
        mapping from global state -> choice for each player, for each strategy
    winner: Identity | None
        winner of game
    """
    def __init__(
        self,
        *args: dict[str, Any] | list[dict[str, Any]],
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
        self.players: list[Player] = [
            Player(
                identity,
                cards=self.cards,
                **(
                    {}
                    if len(args) == 0
                    else args[0]
                    if len(args) == 1
                    else args[identity]
                ),
            )
            for identity in Identity
        ]
        self.players[0].handshake(self.players[1])

        # initialize states
        self.states: list[list[dict[bytes, list[int, list[int]]]]] = [
            [
                {}
                for _ in Network
            ]
            for _ in Identity
        ]
        self.winner: Identity | None = None

    def _play_age(self):
        """Execute an age"""

        # freeze states
        for player in self.players:
            player.freeze_state()

        # play and awaken cards, saving states
        for player, states in zip(self.players, self.states):
            frozen_state = player._frozen_state.tobytes()
            for method, network in [
                (Player.play_cards, Network.PLAY),
                (Player.awaken_card, Network.AWAKEN),
            ]:
                # choose card
                _, choice = method(player)
                (
                    states[network]
                    .setdefault(frozen_state, [])
                    .append(choice)
                )

        # unfreeze states
        for player in self.players:
            player.unfreeze_state()

    def _play_turn(self):
        """Execute a complete turn"""

        # shuffle step (we do this first)
        for player in self.players:
            player.shuffle_cards()
            player.draw_cards(6)

        # play ages
        for _ in range(3):
            self._play_age()

        # battle
        self.players[0].battle_opponent()

    def play(self) -> Game:
        """Play game

        Returns
        -------
        Game
            calling instance
        """
        for _ in range(self._turn_limit):
            self._play_turn()
            for player in self.players:
                if player.temples <= 0:
                    self.winner = player.opponent.identity
                    return self

        # draw
        self.winner = None
        return self
