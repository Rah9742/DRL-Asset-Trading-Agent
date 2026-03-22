"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_return_series(portfolio_values: pd.Series) -> pd.Series:
    """Compute arithmetic returns from a portfolio value series."""
    returns = portfolio_values.pct_change().dropna()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def compute_performance_metrics(portfolio_values: pd.Series, annualization_factor: int = 252) -> dict[str, float]:
    """Calculate standard portfolio metrics for experiment comparison."""
    if portfolio_values.empty:
        raise ValueError("Portfolio value series must not be empty.")

    returns = compute_return_series(portfolio_values)
    cumulative_return = float(portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1.0)

    if returns.empty:
        return {
            "cumulative_return": cumulative_return,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    avg_return = returns.mean()
    volatility = returns.std(ddof=0)
    annualized_return = float((1.0 + avg_return) ** annualization_factor - 1.0)
    annualized_volatility = float(volatility * np.sqrt(annualization_factor))
    sharpe_ratio = 0.0 if annualized_volatility == 0.0 else float(annualized_return / annualized_volatility)
    max_drawdown = float(_max_drawdown(portfolio_values))

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def _max_drawdown(portfolio_values: pd.Series) -> float:
    running_peak = portfolio_values.cummax()
    drawdown = portfolio_values / running_peak - 1.0
    return drawdown.min()
