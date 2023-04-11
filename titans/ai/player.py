"""Player module"""

from __future__ import annotations

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Identity, Name, Network


class Player:
    """Player class

    This class performs all of the core operations of each player.

    Parameters
    ----------
    identity: Identity
        player's identity
    cards: list[Card]
        cards used in this game
    strategies: list[np.ndarray | None]
        strategies for performing actions. If None, random choices will be used
    random_state: int | None, optional, default=None
        player's random seed

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
    ritual_piles: list[Card]
        cards in the shared ritual piles. This is harmonized between players in
        Player.handshake()
    strategies: list[np.ndarray | None]
        strategies for performing actions
    """
    def __init__(
        self,
        identity: Identity,
        /,
        cards: list[Card],
        strategies: list[np.ndarray | None] = [None for _ in Network],
        *,
        random_state: int = None,
    ):
        # save identity and strategies
        self.identity: Identity = identity
        self.strategies: list[np.ndarray] = strategies

        # initialize zones
        self.deck_zone: list[Card] = []
        self.discard_zone: list[Card] = []
        self.hand_zone: list[Card] = []
        self.play_zone: list[Card] = []
        self.ritual_piles: list[Card] = []

        # create deck and ritual piles
        for c, card in enumerate(cards):

            # make starting deck
            if card.name in [Name.MONK, Name.WIZARD]:

                # this player gets every other starting card
                if c % len(Identity) == self.identity:
                    self.deck_zone.append(card)

            # add card to ritual piles
            else:
                self.ritual_piles.append(card)

        # initialize opponent
        self.opponent: Player | None = None

        # initialize rng
        self.rng = np.random.default_rng(random_state)

    def _get_global_state(self) -> np.ndarray:
        """Get your private state + opponent's public state

        Returns
        -------
        np.ndarray
            global state, from this instance's point-of-view
        """

        # make sure opponent is initialized
        if self.opponent is None:
            raise AttributeError(
                "No opponent set!"
                " You must run Player.handshake() before playing!"
            )

        # return state
        return np.concatenate((
            self.get_state(public=False),
            self.opponent.get_state(public=True),
        ))

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
        # constructed public knowledge in state(). This order is also
        # hard-coded into the test test_get_state(); that should be updated at
        # some point.
        return [
            self.discard_zone,
            self.play_zone,
            self.deck_zone,
            self.hand_zone,
        ]

    def awaken_card(self, /):
        """Awaken a card from the ritual piles

        The awakened card is added to your deck. You can always choose to not
        awaken.
        """

    def draw_cards(self, count: int = 1, /) -> list[Card]:
        """Draw cards

        Parameters
        ----------
        count: int
            number of cards to draw

        Returns
        -------
        list[Card]
            the cards drawn. These are added to self.hand_zone, but are also
            returned for easy debugging / logging.
        """
        drawn = []
        for _ in range(count):

            # if we run out of cards to draw, then stop early
            if not self.deck_zone:
                return drawn

            # draw card, add to hand
            drawn.append(self.deck_zone.pop())
            self.hand_zone.append(drawn[-1])

        # return list of drawn cards
        return drawn

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
            zero_state = not public or zone is not self.hand_zone
            save_state = not public or zone is not self.deck_zone
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

    def handshake(self, opponent: Player, /):
        """Set this player's opponent (and vice-versa)

        This sets self.opponent, which is used when getting the global game
        state

        You always must handshake before starting a game.

        Parameters
        ----------
        opponent: Player
            this instance's competitor for the game
        """
        self.opponent = opponent
        opponent.opponent = self
        opponent.ritual_piles = self.ritual_piles

    def play_cards(self, count: int = 1, /) -> list[Card]:
        """Play one or more cards

        Parameters
        ----------
        count: int, optional, default=1
            how many cards to play

        Returns
        -------
        list[Card]
            cards played. These are automatically added to self.play_zone, but
            are also returned for easy debugging / logging
        """

        # initialize list of cards played
        played = []

        # play multiple
        if count > 1:
            for _ in range(count):
                played.extend(self.play_cards())
            return played

        # get decision matrix
        decision_matrix = (
            self._get_global_state() @ self.strategies[Network.PLAY]
            if self.strategies[Network.PLAY] is not None
            else self.rng.random(len(Name) + 1)
        )

        # play highest-valued card that we can play
        name_list = np.array([card.name for card in self.hand_zone])
        for choice in np.argsort(decision_matrix)[::-1]:

            # play top card of deck
            if choice == len(Name) and len(self.deck_zone) > 0:
                card = self.deck_zone.pop()
                played.append(card)
                self.play_zone.append(card)
                break

            # play card from hand
            if (matches := np.argwhere(name_list == choice)).any():
                card = self.hand_zone.pop(matches[0][0])
                played.append(card)
                self.play_zone.append(card)
                break

        # return card played
        return played

    def shuffle_cards(self):
        """Shuffle all cards together"""

        # move all cards to the deck
        self.deck_zone += self.discard_zone + self.hand_zone + self.play_zone
        self.discard_zone.clear()
        self.hand_zone.clear()
        self.play_zone.clear()

        # shuffle order
        self.rng.shuffle(self.deck_zone)
