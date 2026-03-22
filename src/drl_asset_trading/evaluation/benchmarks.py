"""Benchmark runners for heuristic trading strategies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import EnvironmentConfig, ExperimentConfig
from ..data import split_by_dates
from ..envs import TradingEnvironment
from ..strategies import BuyAndHoldStrategy, RandomStrategy
from .metrics import compute_performance_metrics
from .runner import run_strategy_episode

NON_FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]


def default_processed_dataset_path(dataset_name: str, ticker: str, start_date: str, end_date: str) -> Path:
    """Return the default processed dataset path."""
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    return Path("data/processed") / dataset_name / filename


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    """Load a processed dataset from CSV."""
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def derive_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Derive the state feature columns from a processed dataset."""
    return [column for column in dataset.columns if column not in NON_FEATURE_COLUMNS]


def run_benchmark_suite(dataset: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run heuristic benchmarks across train, validation, and test splits."""
    feature_columns = derive_feature_columns(dataset)
    splits = split_by_dates(dataset, config.splits)
    strategies = {
        "buy_and_hold": BuyAndHoldStrategy(),
        "random": RandomStrategy(seed=config.experiment.random_seed),
    }

    metrics_rows: list[dict[str, object]] = []
    history_frames: dict[str, pd.DataFrame] = {}

    for split_name, split_frame in splits.items():
        for strategy_name, strategy in strategies.items():
            environment = TradingEnvironment(
                market_data=split_frame,
                feature_columns=feature_columns,
                config=config.environment,
            )
            history = run_strategy_episode(strategy, environment)
            metrics = compute_performance_metrics(
                history["portfolio_value"],
                annualization_factor=config.environment.annualization_factor,
            )

            metrics_rows.append(
                {
                    "dataset": split_name,
                    "strategy": strategy_name,
                    **metrics,
                }
            )
            history_frames[f"{split_name}_{strategy_name}"] = history

    return pd.DataFrame(metrics_rows), history_frames


def save_benchmark_outputs(
    metrics: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    dataset_name: str,
    ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[Path, list[Path]]:
    """Persist benchmark metrics and episode histories."""
    stem = f"{ticker}_{start_date}_{end_date}"
    metrics_path = Path("results") / dataset_name / f"{stem}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    history_paths: list[Path] = []
    history_dir = Path("results") / dataset_name / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)

    for history_name, history in histories.items():
        history_path = history_dir / f"{stem}_{history_name}.csv"
        history.to_csv(history_path, index=False)
        history_paths.append(history_path)

    return metrics_path, history_paths
