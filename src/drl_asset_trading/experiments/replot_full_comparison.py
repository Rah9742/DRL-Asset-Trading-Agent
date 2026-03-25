"""Rebuild report plots from existing full-comparison history artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ..config import ExperimentConfig
from ..evaluation import plot_drawdowns, plot_equity_curves, plot_equity_curves_with_variance
from .run_ablation import configure_experiment

EXPERIMENT_SETTINGS = [
    ("profit", "none"),
    ("profit", "sparse"),
    ("profit", "decay"),
    ("sharpe", "none"),
    ("sharpe", "sparse"),
    ("sharpe", "decay"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the replot runner."""
    parser = argparse.ArgumentParser(description="Rebuild full-comparison plots from existing history CSV files.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list overriding experiment.seed_values.",
    )
    return parser.parse_args()


def replot_full_comparison(config: ExperimentConfig, seeds: list[int] | None = None) -> dict[str, Path]:
    """Recreate the report plots from saved test histories without rerunning training."""
    base_config = deepcopy(config)
    seed_values = seeds or list(base_config.experiment.seed_values)
    stem = f"{base_config.data.ticker}_{base_config.data.start_date}_{base_config.data.end_date}"
    report_dir = Path("reports") / base_config.data.ticker
    test_range_label = f"{base_config.splits.validation_end} to {base_config.splits.test_end}"

    test_histories_for_plots: dict[str, pd.DataFrame] = {
        "buy_and_hold": _load_history(
            report_dir / "heuristics" / "heuristic_reference" / "histories" / f"{stem}_test_buy_and_hold.csv"
        ),
        "random": _load_history(
            report_dir / "heuristics" / "heuristic_reference" / "histories" / f"{stem}_test_random.csv"
        ),
    }
    test_history_groups_for_plots: dict[str, list[pd.DataFrame]] = {
        "buy_and_hold": [test_histories_for_plots["buy_and_hold"]],
        "random": [test_histories_for_plots["random"]],
    }

    for reward_mode, sentiment_variant in EXPERIMENT_SETTINGS:
        variant_config = deepcopy(config)
        variant_config, _, run_name = configure_experiment(
            config=variant_config,
            reward_mode=reward_mode,
            sentiment_variant=sentiment_variant,
        )
        seed_histories = []
        for seed in seed_values:
            history_path = (
                Path(variant_config.rl.results_dir)
                / f"{run_name}_seed{seed}"
                / f"{stem}_test_history.csv"
            )
            seed_histories.append(_load_history(history_path))

        plot_key = f"{run_name}_dqn"
        test_history_groups_for_plots[plot_key] = seed_histories
        test_histories_for_plots[plot_key] = _average_test_history(seed_histories)

    equity_plot_path = plot_equity_curves(
        histories=test_histories_for_plots,
        path=report_dir / f"{stem}_test_equity_curves.png",
        title=f"{base_config.data.ticker} Test Split Equity Curves ({test_range_label})",
    )
    equity_variance_plot_path = plot_equity_curves_with_variance(
        history_groups=test_history_groups_for_plots,
        path=report_dir / f"{stem}_test_equity_curves_with_variance.png",
        title=f"{base_config.data.ticker} Test Split Equity Curves with Seed Variance ({test_range_label})",
    )
    drawdown_plot_path = plot_drawdowns(
        histories=test_histories_for_plots,
        path=report_dir / f"{stem}_test_drawdowns.png",
        title=f"{base_config.data.ticker} Test Split Drawdowns ({test_range_label})",
    )
    return {
        "equity_plot_path": equity_plot_path,
        "equity_variance_plot_path": equity_variance_plot_path,
        "drawdown_plot_path": drawdown_plot_path,
    }


def _load_history(path: Path) -> pd.DataFrame:
    """Load a saved history CSV and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required history file not found: {path}")
    return pd.read_csv(path)


def _average_test_history(histories: list[pd.DataFrame]) -> pd.DataFrame:
    """Average portfolio values across test histories by date."""
    if not histories:
        raise ValueError("At least one history is required to compute an average test history.")

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
    return pd.DataFrame(
        {
            "date": merged["date"],
            "portfolio_value": merged[value_columns].mean(axis=1),
        }
    )


def _parse_seed_override(seed_text: str) -> list[int]:
    """Parse a comma-separated seed list."""
    return [int(chunk.strip()) for chunk in seed_text.split(",") if chunk.strip()]


def main() -> None:
    """Rebuild plots from existing history CSV files."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    seeds = _parse_seed_override(args.seeds) if args.seeds else None
    paths = replot_full_comparison(config=config, seeds=seeds)
    print(f"Equity plot: {paths['equity_plot_path']}")
    print(f"Equity variance plot: {paths['equity_variance_plot_path']}")
    print(f"Drawdown plot: {paths['drawdown_plot_path']}")


if __name__ == "__main__":
    main()
