"""Lightweight plotting utilities for equity curves and drawdowns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def plot_equity_curves_with_variance(
    history_groups: dict[str, list[pd.DataFrame]],
    path: str | Path,
    title: str,
) -> Path:
    """Plot mean equity curves with a shaded one-standard-deviation band."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (label, histories) in enumerate(history_groups.items()):
        dates, mean_values, std_values = _aggregate_history_group(histories)
        color = colors[index % len(colors)]
        plt.plot(dates, mean_values, label=label, color=color)
        if len(histories) > 1:
            lower = mean_values - std_values
            upper = mean_values + std_values
            plt.fill_between(dates, lower, upper, color=color, alpha=0.2)

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


def _aggregate_history_group(histories: list[pd.DataFrame]) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Align a collection of histories by date and compute mean and standard deviation."""
    if not histories:
        raise ValueError("At least one history is required to plot a variance band.")

    merged = None
    for index, history in enumerate(histories):
        frame = history.loc[:, ["date", "portfolio_value"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.rename(columns={"portfolio_value": f"portfolio_value_{index}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="date", how="inner")

    value_columns = [column for column in merged.columns if column.startswith("portfolio_value_")]
    values = merged.loc[:, value_columns]
    mean_values = values.mean(axis=1).to_numpy(dtype=float)
    std_values = values.std(axis=1, ddof=0).to_numpy(dtype=float)
    return merged["date"], mean_values, std_values
