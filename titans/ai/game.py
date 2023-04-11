"""Game Module"""

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Name, Identity
from titans.ai.player import Player


class Game:
    """Game Class

    Parameters
    ----------
    strategies: list[None | list[np.ndarray | None]] | None, optional, default=None
        players' strategies
    """  # noqa
    def __init__(
        self,
        strategies: list[None | list[np.ndarray | None]] | None = None,
    ):
        # default strategies to random
        if strategies is None:
            strategies = [None for _ in Identity]

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
                strategies=strategy,
            )
            for identity, strategy in zip(Identity, strategies)
        ]
        self.players[0].handshake(self.players[1])
