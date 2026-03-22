"""Single-asset trading environment skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import EnvironmentConfig


@dataclass(slots=True)
class TradingStep:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class TradingEnvironment:
    """Long/flat single-asset environment with a discrete action space."""

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, market_data: pd.DataFrame, feature_columns: list[str], config: EnvironmentConfig) -> None:
        if market_data.empty:
            raise ValueError("Market data must not be empty.")
        self.market_data = market_data.reset_index(drop=False).copy()
        self.feature_columns = feature_columns
        self.config = config
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to the first tradable timestep."""
        self.current_step = 0
        self.cash = self.config.initial_cash
        self.units_held = 0.0
        self.position = 0
        self.prev_portfolio_value = self.config.initial_cash
        self.history: list[dict[str, float | int]] = []
        return self._observation()

    def step(self, action: int) -> TradingStep:
        """Advance the environment by one timestep."""
        self._validate_action(action)
        price = float(self.market_data.loc[self.current_step, "Close"])

        if action == self.BUY and self.position == 0:
            self._buy(price)
        elif action == self.SELL and self.position == 1:
            self._sell(price)

        portfolio_value = self._portfolio_value(price)
        reward = self._compute_reward(portfolio_value)
        self._record_step(action, price, portfolio_value, reward)

        self.current_step += 1
        done = self.current_step >= len(self.market_data)
        observation = self._observation(done=done)
        info = {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "units_held": self.units_held,
            "position": self.position,
        }
        self.prev_portfolio_value = portfolio_value
        return TradingStep(observation=observation, reward=reward, done=done, info=info)

    def _observation(self, done: bool = False) -> np.ndarray:
        """Return the current feature vector with the position appended."""
        if done:
            return np.zeros(len(self.feature_columns) + 1, dtype=float)

        values = self.market_data.loc[self.current_step, self.feature_columns].to_numpy(dtype=float)
        return np.append(values, float(self.position))

    def _buy(self, price: float) -> None:
        tradable_cash = self.cash * (1.0 - self.config.transaction_cost)
        self.units_held = tradable_cash / price
        self.cash = 0.0
        self.position = 1

    def _sell(self, price: float) -> None:
        gross_value = self.units_held * price
        self.cash = gross_value * (1.0 - self.config.transaction_cost)
        self.units_held = 0.0
        self.position = 0

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.units_held * price

    def _compute_reward(self, portfolio_value: float) -> float:
        """Support a simple return reward and a lightweight risk-aware placeholder."""
        step_return = (portfolio_value / self.prev_portfolio_value) - 1.0
        if self.config.reward_mode == "profit":
            return float(step_return)
        if self.config.reward_mode == "risk_adjusted":
            penalty = abs(step_return) * 0.1
            return float(step_return - penalty)
        raise ValueError(f"Unsupported reward mode: {self.config.reward_mode}")

    def _record_step(self, action: int, price: float, portfolio_value: float, reward: float) -> None:
        date_value = self.market_data.loc[self.current_step, self.market_data.columns[0]]
        self.history.append(
            {
                "step": self.current_step,
                "date": date_value,
                "action": action,
                "price": price,
                "position": self.position,
                "portfolio_value": portfolio_value,
                "reward": reward,
            }
        )

    @staticmethod
    def _validate_action(action: int) -> None:
        if action not in {0, 1, 2}:
            raise ValueError(f"Invalid action '{action}'. Expected one of 0, 1, 2.")
