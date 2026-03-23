"""Run eta sweeps for differential-Sharpe ablation variants."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ..agents.training import train_double_dqn
from ..config import ExperimentConfig
from ..evaluation.benchmarks import default_processed_dataset_path, load_processed_dataset
from .run_ablation import configure_experiment

ETA_SWEEP_SENTIMENT_VARIANTS = ("none", "decay")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the eta sweep runner."""
    parser = argparse.ArgumentParser(description="Sweep differential Sharpe eta values across DSR variants.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--sentiment-variants",
        nargs="+",
        choices=sorted(("none", "zero", "decay")),
        default=list(ETA_SWEEP_SENTIMENT_VARIANTS),
        help="Sentiment variants to include in the eta sweep.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional seed list overriding experiment.seed_values.",
    )
    parser.add_argument(
        "--etas",
        nargs="*",
        type=float,
        default=None,
        help="Optional eta grid overriding environment.differential_sharpe_eta_candidates.",
    )
    return parser.parse_args()


def run_differential_sharpe_eta_sweep(
    config: ExperimentConfig,
    sentiment_variants: list[str] | None = None,
    seeds: list[int] | None = None,
    etas: list[float] | None = None,
) -> dict[str, object]:
    """Train differential-Sharpe variants across an eta grid and summarize test performance."""
    base_config = deepcopy(config)
    run_sentiment_variants = sentiment_variants or list(ETA_SWEEP_SENTIMENT_VARIANTS)
    seed_values = seeds or list(base_config.experiment.seed_values)
    eta_values = etas or list(base_config.environment.differential_sharpe_eta_candidates)

    report_dir = Path("reports") / base_config.data.ticker / "eta_sweeps"
    report_dir.mkdir(parents=True, exist_ok=True)

    seed_metric_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for sentiment_variant in run_sentiment_variants:
        variant_config = deepcopy(base_config)
        variant_config, dataset_name, run_name = configure_experiment(
            config=variant_config,
            reward_mode="sharpe",
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
                f"Processed dataset not found at {dataset_path}. Build features before running the eta sweep."
            )

        dataset = load_processed_dataset(dataset_path)
        print(
            f"Running eta sweep for experiment '{run_name}' "
            f"(sentiment_variant={variant_config.features.sentiment_variant}, "
            f"etas={eta_values}, seeds={seed_values})",
            flush=True,
        )

        for eta in eta_values:
            eta_seed_frames: list[pd.DataFrame] = []
            eta_label = _format_eta_label(eta)

            for seed in seed_values:
                seed_config = deepcopy(variant_config)
                seed_config.experiment.random_seed = seed
                seed_config.environment.differential_sharpe_eta = eta
                seed_run_name = f"{run_name}_{eta_label}_seed{seed}"
                print(f"  -> eta {eta:.6f}, seed {seed}", flush=True)

                ddqn_results = train_double_dqn(
                    dataset=dataset,
                    config=seed_config,
                    dataset_name=dataset_name,
                    run_name=seed_run_name,
                )

                ddqn_metrics = ddqn_results["split_metrics"].copy()
                ddqn_metrics["comparison_group"] = run_name
                ddqn_metrics["dataset_name"] = dataset_name
                eta_seed_frames.append(ddqn_metrics)

            eta_metrics = pd.concat(eta_seed_frames, ignore_index=True, sort=False)
            seed_metric_frames.append(eta_metrics)

            test_only = eta_metrics.loc[eta_metrics["dataset"] == "test"].copy()
            summary_rows.append(
                {
                    "comparison_group": run_name,
                    "reward_mode": variant_config.environment.reward_mode,
                    "sentiment_variant": variant_config.features.sentiment_variant,
                    "dataset_name": dataset_name,
                    "differential_sharpe_eta": eta,
                    "seed_count": len(seed_values),
                    "test_cumulative_return_mean": test_only["cumulative_return"].mean(),
                    "test_cumulative_return_std": test_only["cumulative_return"].std(ddof=0),
                    "test_annualized_return_mean": test_only["annualized_return"].mean(),
                    "test_annualized_return_std": test_only["annualized_return"].std(ddof=0),
                    "test_annualized_volatility_mean": test_only["annualized_volatility"].mean(),
                    "test_annualized_volatility_std": test_only["annualized_volatility"].std(ddof=0),
                    "test_sharpe_ratio_mean": test_only["sharpe_ratio"].mean(),
                    "test_sharpe_ratio_std": test_only["sharpe_ratio"].std(ddof=0),
                    "test_max_drawdown_mean": test_only["max_drawdown"].mean(),
                    "test_max_drawdown_std": test_only["max_drawdown"].std(ddof=0),
                }
            )

    seed_metrics = pd.concat(seed_metric_frames, ignore_index=True, sort=False)
    summary = pd.DataFrame(summary_rows).sort_values(
        by=["comparison_group", "differential_sharpe_eta"],
        ignore_index=True,
    )

    stem = f"{base_config.data.ticker}_{base_config.data.start_date}_{base_config.data.end_date}"
    seed_metrics_path = report_dir / f"{stem}_differential_sharpe_eta_seed_metrics.csv"
    summary_path = report_dir / f"{stem}_differential_sharpe_eta_summary.csv"
    seed_metrics.to_csv(seed_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Eta seed metrics: {seed_metrics_path}", flush=True)
    print(f"Eta summary: {summary_path}", flush=True)
    print(summary.to_string(index=False))
    return {
        "seed_metrics": seed_metrics,
        "seed_metrics_path": seed_metrics_path,
        "summary": summary,
        "summary_path": summary_path,
        "report_dir": report_dir,
    }


def _format_eta_label(eta: float) -> str:
    """Convert an eta value into a filesystem-safe label."""
    eta_text = format(eta, ".6f").rstrip("0").rstrip(".")
    return f"eta{eta_text.replace('.', 'p')}"


def main() -> None:
    """Run the configured differential-Sharpe eta sweep."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    run_differential_sharpe_eta_sweep(
        config=config,
        sentiment_variants=args.sentiment_variants,
        seeds=args.seeds,
        etas=args.etas,
    )


if __name__ == "__main__":
    main()
