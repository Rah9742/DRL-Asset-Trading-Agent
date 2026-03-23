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
        self.dsr_mean_return = 0.0
        self.dsr_mean_squared_return = 0.0
        self.dsr_steps = 0
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
        """Support profit and differential-Sharpe rewards."""
        step_return = (portfolio_value / self.prev_portfolio_value) - 1.0
        if self.config.reward_mode == "profit":
            return float(step_return)
        if self.config.reward_mode == "differential_sharpe":
            return float(self._differential_sharpe_reward(step_return))
        raise ValueError(f"Unsupported reward mode: {self.config.reward_mode}")

    def _differential_sharpe_reward(self, step_return: float) -> float:
        """Compute an online differential Sharpe reward using exponentially weighted moments."""
        variance = self.dsr_mean_squared_return - self.dsr_mean_return ** 2
        # Fall back to plain return until the running moments are both warmed up and numerically stable.
        use_plain_return = (
            self.dsr_steps < self.config.differential_sharpe_warmup_steps
            or variance < self.config.differential_sharpe_min_variance
        )

        if use_plain_return:
            reward = step_return
        else:
            variance = max(variance, self.config.differential_sharpe_epsilon)
            numerator = (
                self.dsr_mean_squared_return * (step_return - self.dsr_mean_return)
                - 0.5 * self.dsr_mean_return * (step_return ** 2 - self.dsr_mean_squared_return)
            )
            reward = numerator / (variance ** 1.5)

        eta = self.config.differential_sharpe_eta
        self.dsr_mean_return += eta * (step_return - self.dsr_mean_return)
        self.dsr_mean_squared_return += eta * (step_return ** 2 - self.dsr_mean_squared_return)
        self.dsr_steps += 1
        return reward

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
