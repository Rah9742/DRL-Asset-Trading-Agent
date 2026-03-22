"""Strategy interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseStrategy(ABC):
    """Minimal policy interface shared by heuristic and RL agents."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal strategy state before a new episode."""

    @abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """Return a discrete trading action for the current observation."""
