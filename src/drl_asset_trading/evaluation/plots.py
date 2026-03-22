"""Lightweight plotting utilities for equity curves and drawdowns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(histories: dict[str, pd.DataFrame], path: str | Path, title: str) -> Path:
    """Plot portfolio value series for a collection of strategy histories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for label, history in histories.items():
        plt.plot(pd.to_datetime(history["date"]), history["portfolio_value"], label=label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_drawdowns(histories: dict[str, pd.DataFrame], path: str | Path, title: str) -> Path:
    """Plot drawdown series for a collection of strategy histories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for label, history in histories.items():
        values = history["portfolio_value"]
        drawdown = values / values.cummax() - 1.0
        plt.plot(pd.to_datetime(history["date"]), drawdown, label=label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
