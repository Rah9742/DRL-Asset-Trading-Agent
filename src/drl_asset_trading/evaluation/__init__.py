"""Evaluation helpers for trading experiments."""

from .benchmarks import load_processed_dataset, run_benchmark_suite, save_benchmark_outputs
from .metrics import compute_performance_metrics, compute_return_series
from .plots import plot_drawdowns, plot_equity_curves, plot_equity_curves_with_variance
from .runner import run_strategy_episode

__all__ = [
    "compute_performance_metrics",
    "compute_return_series",
    "load_processed_dataset",
    "plot_drawdowns",
    "plot_equity_curves",
    "plot_equity_curves_with_variance",
    "run_benchmark_suite",
    "run_strategy_episode",
    "save_benchmark_outputs",
]
