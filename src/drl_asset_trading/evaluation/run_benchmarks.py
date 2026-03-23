"""Run heuristic benchmark strategies on processed datasets."""

from __future__ import annotations

import argparse

from ..config import ExperimentConfig
from .benchmarks import (
    default_processed_dataset_path,
    load_processed_dataset,
    run_benchmark_suite,
    save_benchmark_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run heuristic benchmark strategies.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--dataset",
        choices=["price", "price_sentiment_sparse", "price_sentiment_decay"],
        default="price",
        help="Processed dataset variant to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run buy-and-hold and random benchmarks on the selected dataset."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    dataset_path = default_processed_dataset_path(
        dataset_name=args.dataset,
        ticker=config.data.ticker,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. Build features before running benchmarks."
        )

    dataset = load_processed_dataset(dataset_path)
    metrics, histories, scaler = run_benchmark_suite(dataset, config)
    metrics_path, history_paths, scaler_path, manifest_path = save_benchmark_outputs(
        metrics=metrics,
        histories=histories,
        scaler=scaler,
        dataset_name=args.dataset,
        ticker=config.data.ticker,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Manifest saved: {manifest_path}")
    print("Metrics summary:")
    print(metrics.to_string(index=False))
    print(f"History files saved: {len(history_paths)}")


if __name__ == "__main__":
    main()
