"""Compare profit-reward experiments across sentiment variants."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ..agents.training import train_double_dqn
from ..artifacts import experiment_manifest, write_json_artifact
from ..config import ExperimentConfig
from ..evaluation.benchmarks import default_processed_dataset_path, load_processed_dataset
from .run_ablation import configure_experiment

DEFAULT_SENTIMENT_VARIANTS = ("none", "sparse", "decay")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sentiment-variant comparison runner."""
    parser = argparse.ArgumentParser(description="Compare profit-reward DDQN runs across sentiment variants.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--sentiment-variants",
        nargs="+",
        choices=["none", "sparse", "decay"],
        default=list(DEFAULT_SENTIMENT_VARIANTS),
        help="Sentiment variants to compare under profit reward.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list overriding experiment.seed_values.",
    )
    return parser.parse_args()


def run_profit_sentiment_comparison(
    config: ExperimentConfig,
    sentiment_variants: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, object]:
    """Train profit-reward DDQN runs for multiple sentiment variants and summarize results."""
    base_config = deepcopy(config)
    run_sentiment_variants = sentiment_variants or list(DEFAULT_SENTIMENT_VARIANTS)
    seed_values = seeds or list(base_config.experiment.seed_values)

    report_dir = Path("reports") / base_config.data.ticker / "profit_sentiment_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for sentiment_variant in run_sentiment_variants:
        variant_config = deepcopy(base_config)
        variant_config, dataset_name, run_name = configure_experiment(
            config=variant_config,
            reward_mode="profit",
            sentiment_variant=sentiment_variant,
        )
        dataset_path = default_processed_dataset_path(
            dataset_name=dataset_name,
            ticker=variant_config.data.ticker,
            start_date=variant_config.data.start_date,
            end_date=variant_config.data.end_date,
        )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {dataset_path}. Build features before running the comparison."
            )

        dataset = load_processed_dataset(dataset_path)
        print(
            f"Running experiment '{run_name}' "
            f"(sentiment_variant={sentiment_variant}, seeds={seed_values})",
            flush=True,
        )

        per_seed_metrics: list[pd.DataFrame] = []
        for seed in seed_values:
            seed_config = deepcopy(variant_config)
            seed_config.experiment.random_seed = seed
            seed_run_name = f"{run_name}_seed{seed}"
            print(f"  -> seed {seed}", flush=True)

            ddqn_results = train_double_dqn(
                dataset=dataset,
                config=seed_config,
                dataset_name=dataset_name,
                run_name=seed_run_name,
            )

            ddqn_metrics = ddqn_results["split_metrics"].copy()
            ddqn_metrics["comparison_group"] = run_name
            per_seed_metrics.append(ddqn_metrics)

        variant_metrics = pd.concat(per_seed_metrics, ignore_index=True, sort=False)
        comparison_rows.append(variant_metrics)

        test_only = variant_metrics.loc[variant_metrics["dataset"] == "test"].copy()
        summary_rows.append(
            {
                "comparison_group": run_name,
                "reward_mode": "profit",
                "sentiment_variant": sentiment_variant,
                "seed_count": len(seed_values),
                "test_sharpe_ratio_mean": test_only["sharpe_ratio"].mean(),
                "test_sharpe_ratio_std": test_only["sharpe_ratio"].std(ddof=0),
                "test_annualized_return_mean": test_only["annualized_return"].mean(),
                "test_annualized_return_std": test_only["annualized_return"].std(ddof=0),
                "test_max_drawdown_mean": test_only["max_drawdown"].mean(),
                "test_max_drawdown_std": test_only["max_drawdown"].std(ddof=0),
            }
        )

    comparison_table = pd.concat(comparison_rows, ignore_index=True, sort=False)
    summary_table = pd.DataFrame(summary_rows).sort_values(by="comparison_group", ignore_index=True)

    stem = f"{base_config.data.ticker}_{base_config.data.start_date}_{base_config.data.end_date}"
    comparison_path = report_dir / f"{stem}_profit_sentiment_comparison.csv"
    summary_path = report_dir / f"{stem}_profit_sentiment_summary.csv"
    comparison_table.to_csv(comparison_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    manifest_path = write_json_artifact(
        report_dir / f"{stem}_manifest.json",
        experiment_manifest(
            base_config,
            dataset_name="profit_sentiment_comparison",
            run_name="profit_sentiment_comparison",
            extra={
                "comparison_path": comparison_path,
                "summary_path": summary_path,
                "seed_values": seed_values,
                "sentiment_variants": run_sentiment_variants,
            },
        ),
    )

    print(f"Comparison table: {comparison_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(summary_table.to_string(index=False))
    return {
        "comparison_table": comparison_table,
        "comparison_path": comparison_path,
        "summary_table": summary_table,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "report_dir": report_dir,
    }


def _parse_seed_override(seed_text: str) -> list[int]:
    """Parse a comma-separated seed list."""
    return [int(chunk.strip()) for chunk in seed_text.split(",") if chunk.strip()]


def main() -> None:
    """Run the profit-reward sentiment comparison."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    seeds = _parse_seed_override(args.seeds) if args.seeds else None
    run_profit_sentiment_comparison(
        config=config,
        sentiment_variants=args.sentiment_variants,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
