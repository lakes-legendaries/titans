"""Player module"""

from __future__ import annotations

import numpy as np

from titans.ai.card import Card
from titans.ai.enum import Ability, Action, Identity, Name, Zone
from titans.ai.strategy import RandomStrategy, Strategy


class Player:
    """Player class

    This class performs all of the core operations of each player.

    Parameters
    ----------
    identity: Identity
        player's identity
    cards: list[Card]
        cards used in this game
    strategies: dict[Action, Strategy] | None, optional, default=None
        strategies for performing actions. If None, random choices will be used
    random_state: int | None, optional, default=None
        player's random seed
    temperature: float | None, optional, default=None
        randomness to add into decision making

    Attributes
    ----------
    cards: dict[Zone, list[Card]]
        player's cards, in each zone
    identity: Identity
        player's identity
    ritual_piles: list[Card]
        cards in the shared ritual piles. This is harmonized between players in
        Player.handshake()
    strategies: dict[Action, Strategy]
        strategies for performing actions
    temples: int
        number of temples under player's control
    """
    def __init__(
        self,
        identity: Identity,
        /,
        cards: list[Card],
        strategies: dict[Action, Strategy] | None = None,
        *,
        random_state: int | None = None,
        temperature: float | None = None,
    ):
        # save identity and strategies
        self.identity: Identity = identity
        self.strategies: dict[Action, Strategy] = (
            strategies
            if strategies is not None
            else {
                action: RandomStrategy(random_state=random_state)
                for action in Action
            }
        )
        self._temperature = temperature

        # initialize player's card zones and temples
        self.temples: int = 3
        self.cards: dict[Zone: list[card]] = {
            zone: []
            for zone in Zone
        }

        # create deck and ritual piles
        self.ritual_piles: list[Card] = []
        for c, card in enumerate(cards):

            # make starting deck
            if card.name in [Name.MONK, Name.WIZARD]:

                # this player gets every other starting card
                if c % len(Identity) == self.identity.value:
                    self.cards[Zone.DECK].append(card)

            # add card to ritual piles
            else:
                self.ritual_piles.append(card)

        # initialize placeholders
        self.opponent: Player | None = None
        self._frozen_state: np.ndarray | None = None
        self._decision_matrices: dict[Action, np.ndarray] | None = None

        # initialize rng
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    def _add_stochasticity(self, decision_matrix: np.ndarray) -> np.ndarray:
        """Add noise to decision matrix

        Amplitude is decided by temperature

        Parameters
        ----------
        decision_matrix: np.ndarray
            input decision matrix

        Returns
        -------
        np.ndarray
            decision matrix with noise applied
        """
        if self._temperature is not None:
            decision_matrix += self._rng.normal(
                scale=self._temperature,
                size=decision_matrix.shape,
            )
        return decision_matrix

    def _get_individual_state(self, public: bool) -> np.ndarray:
        """Get player state

        This is the numeric state that is fed into the ML model as the training
        data (when combining a player's own private state with their opponent's
        public state). The public state is the knowledge your opponent has,
        while the private verstion includes which cards are in your hand.

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
        for zone in Zone:

            # skip hand if public knowledge only
            if public and zone == Zone.HAND:
                continue

            # get counts in zone
            counts = np.zeros(len(Name))
            for card in self.cards[zone]:
                counts[card.name.value] += 1

            # combine hand and deck for public knowledge only
            if public and zone == Zone.DECK:
                for card in self.cards[Zone.HAND]:
                    counts[card.name.value] += 1

            # append counts to state
            state = np.concatenate((state, counts))

        # get overall card counts for each zone. This helps with
        # publicly-available knowledge.
        state = np.concatenate((
            state,
            [
                len(cards)
                for cards in self.cards.values()
            ],
        ))

        # return
        return state

    def awaken_card(self, /) -> tuple[Card | None, int]:
        """Awaken a card from the ritual piles

        The awakened card is added to your discard pile. You can always choose
        to not awaken.

        Returns
        -------
        Card | None
            awakened card. This is automatically added to the discard zone but
            is returned here for easy debugging / logging. None is returned if
            we choose to not awaken.
        int
            decision made. This is the index of the decision matrix that was
            ultimately chosen.
        """

        # get decision matrix
        decision_matrix = self._add_stochasticity((
            self.strategies[Action.AWAKEN].predict(self.get_state())
        ) if self._decision_matrices is None else (
            self._decision_matrices[Action.AWAKEN]
        ))

        # get our current energy
        energy = self.get_energy()

        # awaken highest-valued accessible card we can afford
        name_list = np.array([card.name.value for card in self.ritual_piles])
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
                self.cards[Zone.DISCARD].append(card)
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
            the cards drawn. These are added to the hand zone, but are also
            returned for easy debugging / logging.
        """
        drawn = []
        for _ in range(count):

            # if we run out of cards to draw, then stop early
            if not self.cards[Zone.DECK]:
                return drawn

            # draw card, add to hand
            drawn.append(self.cards[Zone.DECK].pop())
            self.cards[Zone.HAND].append(drawn[-1])

        # return list of drawn cards
        return drawn

    def freeze_state(self):
        """Freeze state (for simultaneous actions)

        This causes self.get_state() to return what the state is when you call
        self.freeze_state(). This will persist until you call
        self.unfreeze_state().
        """
        self._frozen_state = self.get_state()

    def get_energy(self) -> int:
        """Get total energy from all cards in play

        Returns
        -------
        int
            available energy
        """
        energy = 0
        for card in self.cards[Zone.PLAY]:
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
        for card in self.cards[Zone.PLAY]:
            power += card.power
        return power

    def get_state(self) -> np.ndarray:
        """Player's state

        This is the numeric state that is fed into the ML model as the training
        data

        Returns
        -------
        np.ndarray
            state, from this instance's point-of-view
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
            self._get_individual_state(public=False),
            self.opponent._get_individual_state(public=True),
        ))

    def handshake(self, opponent: Player, /):
        """Set this player's opponent (and vice-versa)

        This sets self.opponent, which is used when getting the game state

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
            cards played. The played card is automatically added to the play
            zone, but is also returned here for easy debugging / logging
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
        decision_matrix = self._add_stochasticity((
            self.strategies[Action.PLAY].predict(self.get_state())
        ) if self._decision_matrices is None else (
            self._decision_matrices[Action.PLAY]
        ))

        # play highest-valued card that we can play
        name_list = np.array([
            card.name.value
            for card in self.cards[Zone.HAND]
        ])
        for choice in np.argsort(decision_matrix)[::-1]:

            # play top card of deck
            if choice == len(Name) and len(self.cards[Zone.DECK]) > 0:
                card = self.cards[Zone.DECK].pop()
                played.append(card)
                self.cards[Zone.PLAY].append(card)
                choices.append(choice)
                break

            # play card from hand
            if (matches := np.argwhere(name_list == choice)).any():
                card = self.cards[Zone.HAND].pop(matches[0][0])
                played.append(card)
                self.cards[Zone.PLAY].append(card)
                choices.append(choice)
                break

        # return card played and choice made
        return played, choices

    def shuffle_cards(self):
        """Shuffle all cards together"""

        # move all cards to the deck
        self.cards[Zone.DECK] += (
            self.cards[Zone.DISCARD]
            + self.cards[Zone.HAND]
            + self.cards[Zone.PLAY]
        )

        # clear out other zones
        for zone in Zone:
            if zone != Zone.DECK:
                self.cards[zone].clear()

        # shuffle order
        self._rng.shuffle(self.cards[Zone.DECK])

    def unfreeze_state(self):
        """Unfreeze state

        This ends the hold put on the state initiated by self.freeze_state()
        """
        self._frozen_state = None
