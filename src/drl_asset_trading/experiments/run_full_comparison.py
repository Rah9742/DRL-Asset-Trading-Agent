"""Run all ablation variants and produce one combined comparison report."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ..agents.training import train_double_dqn
from ..config import ExperimentConfig
from ..evaluation import plot_drawdowns, plot_equity_curves, run_benchmark_suite, save_benchmark_outputs
from ..evaluation.benchmarks import default_processed_dataset_path, load_processed_dataset
from .run_ablation import ABLATION_VARIANTS, configure_variant


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full comparison runner."""
    parser = argparse.ArgumentParser(description="Run all ablation variants and build a final comparison table.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--sentiment-imputation-mode",
        choices=["zero", "decay"],
        default="decay",
        help="Sentiment imputation mode used for price_sentiment variants.",
    )
    return parser.parse_args()


def run_full_comparison(
    config: ExperimentConfig,
    sentiment_imputation_mode: str = "decay",
) -> dict[str, object]:
    """Run heuristics and Double DQN across all configured ablation variants."""
    base_config = deepcopy(config)
    comparison_rows: list[pd.DataFrame] = []
    test_histories_for_plots: dict[str, pd.DataFrame] = {}

    report_dir = Path("reports") / base_config.data.ticker
    report_dir.mkdir(parents=True, exist_ok=True)

    heuristic_config = deepcopy(config)
    heuristic_config, heuristic_dataset_name, heuristic_run_name = configure_variant(
        config=heuristic_config,
        variant_name="baseline",
        sentiment_imputation_mode=sentiment_imputation_mode,
    )
    heuristic_dataset_path = default_processed_dataset_path(
        dataset_name=heuristic_dataset_name,
        ticker=heuristic_config.data.ticker,
        start_date=heuristic_config.data.start_date,
        end_date=heuristic_config.data.end_date,
    )
    if not heuristic_dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {heuristic_dataset_path}. Build features before running the full comparison."
        )

    heuristic_dataset = load_processed_dataset(heuristic_dataset_path)
    heuristic_metrics, heuristic_histories = run_benchmark_suite(heuristic_dataset, heuristic_config)
    heuristic_metrics["variant"] = "shared_baselines"
    heuristic_metrics["run_name"] = "heuristic_reference"
    heuristic_metrics["state_mode"] = "price_only"
    heuristic_metrics["reward_mode"] = "profit"
    heuristic_metrics["sentiment_imputation_mode"] = "none"
    heuristic_metrics["dataset_name"] = heuristic_dataset_name
    save_benchmark_outputs(
        metrics=heuristic_metrics,
        histories=heuristic_histories,
        dataset_name="heuristic_reference",
        ticker=heuristic_config.data.ticker,
        start_date=heuristic_config.data.start_date,
        end_date=heuristic_config.data.end_date,
        base_dir=report_dir / "heuristics",
    )
    comparison_rows.append(heuristic_metrics)
    test_histories_for_plots["buy_and_hold"] = heuristic_histories["test_buy_and_hold"]
    test_histories_for_plots["random"] = heuristic_histories["test_random"]

    for variant_name in ABLATION_VARIANTS:
        variant_config = deepcopy(config)
        variant_config, dataset_name, run_name = configure_variant(
            config=variant_config,
            variant_name=variant_name,
            sentiment_imputation_mode=sentiment_imputation_mode,
        )
        dataset_path = default_processed_dataset_path(
            dataset_name=dataset_name,
            ticker=variant_config.data.ticker,
            start_date=variant_config.data.start_date,
            end_date=variant_config.data.end_date,
        )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {dataset_path}. Build features before running the full comparison."
            )

        dataset = load_processed_dataset(dataset_path)
        print(
            f"Running variant '{variant_name}' "
            f"(state_mode={variant_config.features.state_mode}, reward_mode={variant_config.environment.reward_mode}, "
            f"sentiment_imputation_mode={variant_config.features.sentiment_imputation_mode})",
            flush=True,
        )

        ddqn_results = train_double_dqn(
            dataset=dataset,
            config=variant_config,
            dataset_name=dataset_name,
            run_name=run_name,
        )

        ddqn_metrics = ddqn_results["split_metrics"].copy()
        ddqn_metrics["variant"] = variant_name
        ddqn_metrics["dataset_name"] = dataset_name

        comparison_rows.append(ddqn_metrics)

        test_histories_for_plots[f"{run_name}_double_dqn"] = ddqn_results["split_histories"]["test"]

    comparison_table = pd.concat(comparison_rows, ignore_index=True, sort=False)
    comparison_path = report_dir / f"{base_config.data.ticker}_{base_config.data.start_date}_{base_config.data.end_date}_comparison_table.csv"
    comparison_table.to_csv(comparison_path, index=False)

    equity_plot_path = plot_equity_curves(
        histories=test_histories_for_plots,
        path=report_dir / f"{base_config.data.ticker}_{base_config.data.start_date}_{base_config.data.end_date}_test_equity_curves.png",
        title="Test Split Equity Curves",
    )
    drawdown_plot_path = plot_drawdowns(
        histories=test_histories_for_plots,
        path=report_dir / f"{base_config.data.ticker}_{base_config.data.start_date}_{base_config.data.end_date}_test_drawdowns.png",
        title="Test Split Drawdowns",
    )

    print(f"Comparison table: {comparison_path}")
    print(f"Equity plot: {equity_plot_path}")
    print(f"Drawdown plot: {drawdown_plot_path}")
    print(comparison_table.to_string(index=False))
    return {
        "comparison_table": comparison_table,
        "comparison_path": comparison_path,
        "equity_plot_path": equity_plot_path,
        "drawdown_plot_path": drawdown_plot_path,
        "report_dir": report_dir,
    }


def main() -> None:
    """Run heuristics and Double DQN across all configured ablation variants."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    run_full_comparison(
        config=config,
        sentiment_imputation_mode=args.sentiment_imputation_mode,
    )


if __name__ == "__main__":
    main()
