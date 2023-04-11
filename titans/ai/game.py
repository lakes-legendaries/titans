"""Game Module"""

from typing import Any

from titans.ai.card import Card
from titans.ai.enum import Name, Identity
from titans.ai.player import Player


class Game:
    """Game Class

    Parameters
    ----------
    *args: dict[str, Any] | list[dict[str, Any]]
        These dictionaries are unpacked to initialize the players. If one arg
        is provided, then this is used to initialize both players. If two are
        provided, then one is used for each player.
    """
    def __init__(
        self,
        *args: dict[str, Any] | list[dict[str, Any]],
    ):
        # construct cards
        self.cards = []
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
        self.players = [
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
