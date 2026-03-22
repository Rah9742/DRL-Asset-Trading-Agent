"""Train and evaluate a Double DQN agent on a processed dataset."""

from __future__ import annotations

import argparse

from ..config import ExperimentConfig
from ..evaluation.benchmarks import default_processed_dataset_path, load_processed_dataset
from .training import train_double_dqn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Double DQN runner."""
    parser = argparse.ArgumentParser(description="Train and evaluate a Double DQN trading agent.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--dataset",
        choices=["baseline", "sentiment_zero", "augmented"],
        default="baseline",
        help="Processed dataset variant to train on.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the baseline Double DQN agent and save artifacts."""
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
            f"Processed dataset not found at {dataset_path}. Build features before training the agent."
        )

    dataset = load_processed_dataset(dataset_path)
    results = train_double_dqn(dataset=dataset, config=config, dataset_name=args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Best checkpoint: {results['best_checkpoint_path']}")
    print(f"Training log: {results['training_log_path']}")
    print(f"Metrics: {results['metrics_path']}")
    print("Evaluation summary:")
    print(results["split_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
