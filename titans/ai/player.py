"""Players"""

from titans.ai.card import Card
from titans.ai.enum import Identity, Name


class Player:
    """Player class"""
    def __init__(
        self,
        identity: Identity,
        /,
        cards: list[Card],
    ):
        # save identity
        self.identity = identity

        # initialize zones
        self.deck: list[Card] = []
        self.discard: list[Card] = []
        self.hand: list[Card] = []

        # create deck
        for c, card in enumerate(cards):

            # only deal out starting cards
            if card.name not in [Name.MONK, Name.WIZARD]:
                continue

            # this player only gets every other starting card
            if c % len(Identity) != self.identity:
                continue

            # save card
            self.deck.append(card)

    def draw(self, count: int = 1, /):
        """Draw cards

        Parameters
        ----------
        count: int
            number of cards to draw
        """
        for _ in range(count):
            if not self.deck:
                return
            self.hand.append(self.deck.pop())

