"""Players"""

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Identity, Name


class Player:
    """Player class

    This class performs all of the core operations of each player.

    Parameters
    ----------
    identity: Identity
        player's identity
    cards: list[Card]
        cards used in this game

    Attributes
    ----------
    deck_zone: list[Card]
        cards in player's deck
    discard_zone: list[Card]
        cards in player's discard
    hand_zone: list[Card]
        cards in player's hand
    identity: Identity
        player's identity
    play_zone: list[Card]
        cards the player has in play
    """
    def __init__(
        self,
        identity: Identity,
        /,
        cards: list[Card],
    ):
        # save identity
        self.identity: Identity = identity

        # initialize zones
        self.deck_zone: list[Card] = []
        self.discard_zone: list[Card] = []
        self.hand_zone: list[Card] = []
        self.play_zone: list[Card] = []

        # create deck
        for c, card in enumerate(cards):

            # only deal out starting cards
            if card.name not in [Name.MONK, Name.WIZARD]:
                continue

            # this player only gets every other starting card
            if c % len(Identity) != self.identity:
                continue

            # save card
            self.deck_zone.append(card)

    @property
    def _zones(self) -> list[list[Card]]:
        """Get zones, in order this class iterates through them

        Having this as a function is useful for testing

        Returns
        -------
        list[list[Card]]
            zones, with their card contents
        """
        # It's important that hand immediately follows deck: This is used in
        # constructed public knowledge in state()
        return [
            self.discard_zone,
            self.play_zone,
            self.deck_zone,
            self.hand_zone,
        ]

    def draw_cards(self, count: int = 1, /):
        """Draw cards

        Parameters
        ----------
        count: int
            number of cards to draw
        """
        for _ in range(count):
            if not self.deck_zone:
                return
            self.hand_zone.append(self.deck_zone.pop())

    def get_state(self, public: bool) -> np.ndarray:
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

        # get cards in each zone
        for zone in self._zones:

            # initialize counts. For the private state, we zero-out the counts
            # and save the counts every time. For the public state, we combine
            # deck and hand, and so we slectively choose when to zero-out and
            # save the state.
            zero_state = not public or zone != self.hand_zone
            save_state = not public or zone != self.deck_zone
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
        state = np.concatenate((state, [len(zone) for zone in self._zones]))

        # return
        return state


    def shuffle_cards(self):
        """Shuffle all cards together"""
        self.deck_zone += self.discard_zone + self.hand_zone + self.play_zone
        self.discard_zone.clear()
        self.hand_zone.clear()
        self.play_zone.clear()
