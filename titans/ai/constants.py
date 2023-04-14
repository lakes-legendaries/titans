"""Constants module"""

from titans.ai.enum import Name, Zone


NUM_CHOICES: int = len(Name) + 1
"""Number of choices for each action"""

NUM_FEATURES: int = 2 * len(Zone) * (len(Name) + 1) - len(Name)
"""Size of player states, fed as input to ML models"""
