"""Player module"""

from __future__ import annotations

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Ability, Action, Identity, Name


class Player:
    """Player class

    This class performs all of the core operations of each player.

    Parameters
    ----------
    identity: Identity
        player's identity
    cards: list[Card]
        cards used in this game
    strategies: list[np.ndarray | None] | None
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
    temples: int
        number of temples under player's control
    """
    def __init__(
        self,
        identity: Identity,
        /,
        cards: list[Card],
        strategies: list[np.ndarray | None] | None = None,
        *,
        random_state: int = None,
    ):
        # save identity and strategies
        self.identity: Identity = identity
        self.strategies: list[np.ndarray] = (
            strategies
            if strategies is not None
            else [None for _ in Action]
        )

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

        # initialize temples
        self.temples: int = 3

        # initialize rng
        self.rng: np.random.Generator = np.random.default_rng(random_state)

        # initialize frozen state
        self._frozen_state: np.ndarray | None = None

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

        # return frozen state
        if self._frozen_state is not None:
            return self._frozen_state

        # return state
        return np.concatenate((
            self.get_state(public=False),
            self.opponent.get_state(public=True),
        ))

    @classmethod
    def _get_global_state_size(cls) -> int:
        """Get length of global state, for dummy constructions

        Returns
        -------
        int
            length of global state
        """
        players = [Player(identity, []) for identity in Identity]
        players[0].handshake(players[1])
        return len(players[0]._get_global_state())

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
        # hard-coded into several tests: Only change if you have a really good
        # reason to!
        return [
            self.discard_zone,
            self.play_zone,
            self.deck_zone,
            self.hand_zone,
        ]

    def awaken_card(self, /) -> tuple[Card | None, int]:
        """Awaken a card from the ritual piles

        The awakened card is added to your discard pile. You can always choose
        to not awaken.

        Returns
        -------
        Card | None
            awakened card. This is automatically added to self.discard_zone,
            but is returned here for easy debugging / logging. None is returned
            if we choose to not awaken.
        int
            decision made. This is the index of the decision matrix that was
            ultimately chosen.
        """

        # get decision matrix
        decision_matrix = (
            self._get_global_state() @ self.strategies[Action.AWAKEN]
            if self.strategies[Action.AWAKEN] is not None
            else self.rng.random(len(Name) + 1)
        )

        # get our current energy
        energy = self.get_energy()

        # awaken highest-valued accessible card we can afford
        name_list = np.array([card.name for card in self.ritual_piles])
        for choice in np.argsort(decision_matrix)[::-1]:

            # choose to not awaken
            if choice == len(Name):
                return None, choice

            # check if card is in ritual piles
            if (matches := np.argwhere(name_list == choice)).any():

                # get chosen card
                ritual_piles_idx = matches[0][0]
                card = self.ritual_piles[ritual_piles_idx]

                # check if we have enough energy to awaken
                if energy < card.cost:
                    continue

                # awaken card
                self.ritual_piles.pop(ritual_piles_idx)
                self.discard_zone.append(card)
                return card, choice

        # error
        raise RuntimeError("Code error in awakening card")

    def battle_opponent(self) -> Identity | None:
        """Battle opponent

        This should NOT be called for each player, but only once

        Returns
        -------
        Identity | None
            winner of the battle
        """

        # make sure opponent is initialized
        if self.opponent is None:
            raise AttributeError(
                "No opponent set!"
                " You must run Player.handshake() before playing!"
            )

        # get power difference
        power_delta = self.get_power() - self.opponent.get_power()

        # battle is fought to a draw
        if abs(power_delta) < 2:
            return None

        # find winner and loser
        winner = self if power_delta > 0 else self.opponent
        loser = self.opponent if power_delta > 0 else self

        # update temple count
        loser.temples -= 1
        if winner.temples < 2:
            winner.temples += 1

        # return winner
        return winner.identity

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

    def freeze_state(self):
        """Freeze global state (for simultaneous actions)

        This causes self._get_global_state() to return what the state is when
        you call self.freeze_state(). This will persist until you call
        self.unfreeze_state().
        """
        self._frozen_state = self._get_global_state()

    def get_energy(self) -> int:
        """Get total energy from all cards in play

        Returns
        -------
        int
            available energy
        """
        energy = 0
        for card in self.play_zone:
            energy += card.abilities.get(Ability.ENERGY, 0)
        return energy

    def get_power(self) -> int:
        """Get total power from all cards in play

        Returns
        -------
        int
            total power
        """
        power = 0
        for card in self.play_zone:
            power += card.power
        return power

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

    def play_cards(self, count: int = 1, /) -> tuple[list[Card], list[int]]:
        """Play one or more cards

        Parameters
        ----------
        count: int, optional, default=1
            how many cards to play

        Returns
        -------
        list[Card]
            cards played. The played card is automatically added to
            self.play_zone, but is also returned here for easy debugging /
            logging
        list[int]
            decisions made (i.e. the indices of the decision matrix that were
            executed)
        """

        # initialize list of cards played
        played = []
        choices = []

        # play multiple
        if count > 1:
            for _ in range(count):
                rez = self.play_cards()
                played.extend(rez[0])
                choices.extend(rez[0])
            return played, choices

        # get decision matrix
        decision_matrix = (
            self._get_global_state() @ self.strategies[Action.PLAY]
            if self.strategies[Action.PLAY] is not None
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
                choices.append(choice)
                break

            # play card from hand
            if (matches := np.argwhere(name_list == choice)).any():
                card = self.hand_zone.pop(matches[0][0])
                played.append(card)
                self.play_zone.append(card)
                choices.append(choice)
                break

        # return card played and choice made
        return played, choices

    def shuffle_cards(self):
        """Shuffle all cards together"""

        # move all cards to the deck
        self.deck_zone += self.discard_zone + self.hand_zone + self.play_zone
        self.discard_zone.clear()
        self.hand_zone.clear()
        self.play_zone.clear()

        # shuffle order
        self.rng.shuffle(self.deck_zone)

    def unfreeze_state(self):
        """Unfreeze global state

        This ends the hold put on the global state initiated by
        self.freeze_state()
        """
        self._frozen_state = None
