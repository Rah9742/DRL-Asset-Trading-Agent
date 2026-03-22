"""Simple baseline strategies for benchmarking."""

from __future__ import annotations

import random

import numpy as np

from .base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """Buy once at the start of an episode, then hold."""

    def __init__(self) -> None:
        self.has_bought = False

    def reset(self) -> None:
        self.has_bought = False

    def select_action(self, observation: np.ndarray) -> int:
        del observation
        if not self.has_bought:
            self.has_bought = True
            return 1
        return 0


class RandomStrategy(BaseStrategy):
    """Sample uniformly from the discrete action space."""

    def __init__(self, seed: int = 42) -> None:
        self._random = random.Random(seed)

    def reset(self) -> None:
        return None

    def select_action(self, observation: np.ndarray) -> int:
        del observation
        return self._random.choice([0, 1, 2])
