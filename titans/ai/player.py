"""Players"""

import numpy as np

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
        self.play: list[Card] = []

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

    def shuffle(self):
        """Shuffle all cards together"""
        self.deck += self.discard + self.hand + self.play
        self.discard.clear()
        self.hand.clear()
        self.play.clear()

    def state(self, public: bool) -> np.ndarray:
        """Get player state

        This is the numeric state that is fed into the ML model as the training
        data. The public state is the knowledge your opponent has, while the
        private verstion includes which cards are in your hand.

        Parameters
        ----------
        public: bool
            whether to only include public knowledge in the state

        Returns
        -------
        np.ndarray
            numeric representation of player's state
        """
        # initialzie state
        state = np.zeros(0)

        # get cards in each zone. It's important that hand immediately follows
        # deck (for reasons outlined below).
        zones = [self.discard, self.play, self.deck, self.hand]
        for zone in zones:

            # initialize counts. For the private state, we zero-out the counts
            # and save the counts every time. For the public state, we combine
            # deck and hand, and so we slectively choose when to zero-out and
            # save the state.
            zero_state = not public or zone != self.hand
            save_state = not public or zone != self.deck
            if zero_state:
                counts = np.zeros(len(Name))

            # get the counts of each card in the zone
            for card in zone:
                counts[card.name] += 1

            # save the state
            if save_state:
                state = np.concatenate((state, counts))

        # get overall card counts for each zone. This helps with
        # publicly-available knowledge.
        state = np.concatenate((state, [len(zone) for zone in zones]))

        # return
        return state
