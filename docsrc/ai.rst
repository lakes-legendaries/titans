#######
AI Code
#######

********
Overview
********

For this project, I designed an AI engine for two reasons:

#. To control the online demo; and
#. To assist with card evaluation and discovery.

The first point should be obvious (even if it's not yet implemented); while the
second is more subtle: I want to make sure that the cards that are in this game
are all completely balanced against one another, and that there's no
overpowered card in this game. (This can be evaluated by investigating the
weights and win percentages given various strategies as the system plays itself
and learns.)

************
Architecture
************

Each :ref:`~titans.ai.strategy.StandardStrategy` object is a feedforward neural
network that maps from the game's current state (described as the number of
cards in each game :ref:`~titans.ai.enum.Zone`) to the decision that should be
made (e.g. awaken such-and-such card, or choose to do nothing). Each
:ref:`~titans.ai.player.Player` has multiple strategy objects: one for each
type of decision they make (e.g. they'd have one for awakening cards and a
seperate one for playing cards).

When training a strategy, each player starts with randomly-initialized neural
networks (or with an explicit :ref:`~titans.ai.strategy.RandomStrategy` state)
and plays many games against another (virtual) player. During each game, each
time a decision is made, the game state (neural network input) and the chosen
decision (neural network output) is recorded. Additionally, at the completion
of each game, each decision is labeled according to whether the player won or
lost the game. These decisions can then be coalesced into a dataset: Each game
state can be assigned labels for each decision based upon that decision's win
percentage. (Of course, most decisions will be labeled with :code:`np.NaN`,
since they did not occur during the game.)

Using this setup, players can play many games against one another, and have
their strategies updated every so-many-number of games. Players can thus learn
strategies through bootstrapping: As they learn, their opponent learns, and
after playing many games, they'll be left with (hopefully) decent strategies.

********
Examples
********

Vs Random Player
================

As a proof-of-concept, a learning player was set up to play against a random
player. After ~800 games, the learning player won >90% of the time.

Code and results can be found `here
<https://github.com/lakes-legendaries/titans/blob/main/nbs/baseline.ipynb>`_.

Vs Previous-Best
================

Work-in-progress

Card Discovery
==============

Work-in-progress

Bonus: Optimization
===================

A batch decision-making process was implemented to accelerate training. With
it, playtime was accelerated approximately 10x. Results can be found `here
<https://github.com/lakes-legendaries/titans/blob/main/nbs/speed_test.ipynb>`_.

***********
Future Work
***********

The biggest additions I hope to make to the AI package include:

#. Deploying this as part of the demo. I'll stand up a top-performing set of
   strategies in a Fast-API container (similar to the website API bot that's
   working now), and have it accept (likely) JSON input while returning
   integer-decisions and/or probability distributions.

#. Getting card discovery / weighting working. I think a straightforward way of
   showing card strength would be to take a highly-trained strategy, and
   compare win percentages between two players, where one player doesn't have
   access to any given card. If losing access to a single card greatly
   decreases win percentage, that'd be a sign of an overpowered card.

#. Implement the remaining cards in code.

*****
Tests
*****

Tests can be invoked by running

.. code-block:: bash

   pytest tests/titans/ai

(We recommend executing :code:`ai` tests separately from other tests.)

*************
API Reference
*************

Strategies
==========

Strategies are the basic "thinking" (and learning!) units of the game. Think of
them as sklearn-compliant classes for making decisions. The base (abstract)
class is :class:`~titans.ai.strategy.Strategy`, which is inherited by
:class:`~titans.ai.strategy.RandomStrategy` and
:class:`~titans.ai.strategy.StandardStrategy`.

RandomStrategy
--------------

.. autoclass:: titans.ai.strategy.RandomStrategy
   :members:

StandardStrategy
----------------

.. autoclass:: titans.ai.strategy.StandardStrategy
   :members:

Strategy
--------

.. autoclass:: titans.ai.strategy.Strategy
   :members:

Basic game constructs
=====================

Here, we list the basic game constructs that are strung together to simulate
games. :class:`~titans.ai.card.Card` objects are used to represent actual
in-game cards. :class:`~titans.ai.player.Player` objects use decks of cards to
play against each other. :class:`~titans.ai.game.Game` objects orchestrate two
players playing against one another. :class:`~titans.ai.trainer.Trainer`
objects take players through many games to learn how to play the game well.

Card
----

.. autoclass:: titans.ai.card.Card
   :members:

Player
------

.. autoclass:: titans.ai.player.Player
   :members:

Game
----

.. autoclass:: titans.ai.game.Game
   :members:

Trainer
-------

.. autoclass:: titans.ai.trainer.Trainer
   :members:

Constants and enums
===================

Here, we list all the constants and enums that are available for easy-to-read
code references. Please note that most enum members are not themselves
documented, as we assume self-explanability to anyone familiar with the game.

The constants are mostly used internally for sizing neural networks.

Most enums here do **not** resolve to integers (without explicit casting),
which helps catch code errors in mixing them up / supplying them in the wrong
order.

NUM_CHOICES
-----------

.. autodata:: titans.ai.constants.NUM_CHOICES

NUM_FEATURES
------------

.. autodata:: titans.ai.constants.NUM_FEATURES

Ability
-------

.. autoclass:: titans.ai.enum.Ability
   :members:
   :undoc-members:

Action
------

.. autoclass:: titans.ai.enum.Action
   :members:
   :undoc-members:

Element
-------

.. autoclass:: titans.ai.enum.Element
   :members:
   :undoc-members:

Identity
--------

.. autoclass:: titans.ai.enum.Identity
   :members:
   :undoc-members:

Name
----

.. autoclass:: titans.ai.enum.Name
   :members:
   :undoc-members:

Species
-------

.. autoclass:: titans.ai.enum.Species
   :members:
   :undoc-members:

Zone
----

.. autoclass:: titans.ai.enum.Zone
   :members:
   :undoc-members:
