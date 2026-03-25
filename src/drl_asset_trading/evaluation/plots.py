"""Lightweight plotting utilities for equity curves and drawdowns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

PLOT_ORDER = [
    "buy_and_hold",
    "random",
    "profit_none_dqn",
    "profit_sparse_dqn",
    "profit_decay_dqn",
    "sharpe_none_dqn",
    "sharpe_sparse_dqn",
    "sharpe_decay_dqn",
]
PLOT_COLORS = {
    "buy_and_hold": "#111827",
    "random": "#9ca3af",
    "profit_none_dqn": "#2563eb",
    "profit_sparse_dqn": "#0f766e",
    "profit_decay_dqn": "#059669",
    "sharpe_none_dqn": "#d97706",
    "sharpe_sparse_dqn": "#dc2626",
    "sharpe_decay_dqn": "#7c3aed",
}
PLOT_LINESTYLES = {
    "buy_and_hold": "--",
    "random": ":",
    "profit_none_dqn": "-",
    "profit_sparse_dqn": "-",
    "profit_decay_dqn": "-",
    "sharpe_none_dqn": "-",
    "sharpe_sparse_dqn": "-",
    "sharpe_decay_dqn": "-",
}


def plot_equity_curves(histories: dict[str, pd.DataFrame], path: str | Path, title: str) -> Path:
    """Plot portfolio value series for a collection of strategy histories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = _create_report_figure()
    for raw_label, history in _ordered_items(histories):
        axis.plot(
            pd.to_datetime(history["date"]),
            history["portfolio_value"],
            label=_format_plot_label(raw_label),
            color=_color_for_label(raw_label),
            linestyle=_linestyle_for_label(raw_label)
        )
    _style_axis(axis, title=title, ylabel="Portfolio Value", value_kind="portfolio")
    _finalize_figure(figure, axis, output_path)
    return output_path


def plot_equity_curves_with_variance(
    history_groups: dict[str, list[pd.DataFrame]],
    path: str | Path,
    title: str,
) -> Path:
    """Plot mean equity curves with a shaded one-standard-deviation band."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = _create_report_figure()
    for raw_label, histories in _ordered_items(history_groups):
        dates, mean_values, std_values = _aggregate_history_group(histories)
        color = _color_for_label(raw_label)
        axis.plot(
            dates,
            mean_values,
            label=_format_plot_label(raw_label),
            color=color,
            linestyle=_linestyle_for_label(raw_label)
        )
        if len(histories) > 1:
            lower = mean_values - std_values
            upper = mean_values + std_values
            axis.fill_between(dates, lower, upper, color=color, alpha=0.16, linewidth=0.0)

    _style_axis(axis, title=title, ylabel="Portfolio Value", value_kind="portfolio")
    _finalize_figure(figure, axis, output_path)
    return output_path


def plot_drawdowns(histories: dict[str, pd.DataFrame], path: str | Path, title: str) -> Path:
    """Plot drawdown series for a collection of strategy histories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = _create_report_figure()
    for raw_label, history in _ordered_items(histories):
        values = history["portfolio_value"]
        drawdown = values / values.cummax() - 1.0
        axis.plot(
            pd.to_datetime(history["date"]),
            drawdown,
            label=_format_plot_label(raw_label),
            color=_color_for_label(raw_label),
            linestyle=_linestyle_for_label(raw_label)
        )
    _style_axis(axis, title=title, ylabel="Drawdown", value_kind="drawdown")
    _finalize_figure(figure, axis, output_path)
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


def _create_report_figure() -> tuple[plt.Figure, plt.Axes]:
    """Create a clean report-style matplotlib figure."""
    with plt.rc_context(
        {
            "font.size": 11,
            "font.family": "Times New Roman",
            "figure.figsize": (10, 6),
            "figure.dpi": 360,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": True,
            "axes.labelsize": 14,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "legend.facecolor": "white",
            "legend.framealpha": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.edgecolor": "#262626",
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    ):
        figure, axis = plt.subplots()
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    return figure, axis


def _style_axis(axis: plt.Axes, title: str, ylabel: str, value_kind: str) -> None:
    """Apply common report styling to a plot axis."""
    axis.set_title(title, pad=10)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(True)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if value_kind == "portfolio":
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.yaxis.set_minor_locator(MultipleLocator(500))
    elif value_kind == "drawdown":
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.margins(x=0.01)


def _finalize_figure(figure: plt.Figure, axis: plt.Axes, output_path: Path) -> None:
    """Render a polished legend and write the figure to disk."""
    handles, labels = axis.get_legend_handles_labels()
    legend = axis.legend(
        handles,
        labels,
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#cfcfcf",
        title="Strategies",
        title_fontsize=10,
        borderpad=0.5,
        labelspacing=0.35,
        handlelength=2.4,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_linewidth(0.8)
    figure.autofmt_xdate(rotation=0, ha="center")
    for label in axis.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _ordered_items(items: dict[str, object]) -> list[tuple[str, object]]:
    """Return plot items in a stable report-friendly order."""
    order_lookup = {label: index for index, label in enumerate(PLOT_ORDER)}
    return sorted(items.items(), key=lambda pair: (order_lookup.get(pair[0], len(PLOT_ORDER)), pair[0]))


def _color_for_label(raw_label: str) -> str:
    """Return a stable color for a plot label across tickers."""
    return PLOT_COLORS.get(raw_label, "#334155")


def _linestyle_for_label(raw_label: str) -> str:
    """Return a stable line style for a plot label across tickers."""
    return PLOT_LINESTYLES.get(raw_label, "-")


def _format_plot_label(raw_label: str) -> str:
    """Convert internal run labels into report-friendly legend text."""
    if raw_label == "buy_and_hold":
        return "Buy and Hold"
    if raw_label == "random":
        return "Random"
    label = raw_label.replace("_dqn", " ddqn").replace("_", " ")
    words = []
    for word in label.split():
        if word.lower() == "ddqn":
            words.append("DDQN")
        else:
            words.append(word.capitalize())
    return " ".join(words)
