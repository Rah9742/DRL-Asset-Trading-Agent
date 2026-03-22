"""Strategy interfaces and simple baselines."""

from .base import BaseStrategy
from .baseline import BuyAndHoldStrategy, RandomStrategy

__all__ = ["BaseStrategy", "BuyAndHoldStrategy", "RandomStrategy"]
