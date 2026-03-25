"""Run Double DQN experiments using reward-mode and sentiment-variant axes."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..agents.training import train_double_dqn
from ..config import ExperimentConfig, normalize_reward_mode, normalize_sentiment_variant, state_mode_from_sentiment_variant
from ..evaluation.benchmarks import default_processed_dataset_path, load_processed_dataset
from ..features import resolve_processed_dataset_name


def configure_experiment(
    config: ExperimentConfig,
    reward_mode: str,
    sentiment_variant: str,
) -> tuple[ExperimentConfig, str, str]:
    """Apply the reward/sentiment settings and resolve dataset + run names."""
    config.environment.reward_mode = normalize_reward_mode(reward_mode)
    config.features.sentiment_variant = normalize_sentiment_variant(sentiment_variant)
    config.features.sentiment_imputation_mode = config.features.sentiment_variant
    config.features.state_mode = state_mode_from_sentiment_variant(config.features.sentiment_variant)

    dataset_name = resolve_processed_dataset_name(config.features.sentiment_variant)
    run_name = f"{config.environment.reward_mode}_{config.features.sentiment_variant}"
    return config, dataset_name, run_name


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(description="Run a Double DQN experiment.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["profit", "sharpe"],
        required=True,
        help="Reward objective used for training.",
    )
    parser.add_argument(
        "--sentiment-variant",
        choices=["none", "sparse", "decay"],
        required=True,
        help="Sentiment feature variant used to select the processed dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a single Double DQN experiment using the common training pipeline."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    config, dataset_name, run_name = configure_experiment(
        config=config,
        reward_mode=args.reward_mode,
        sentiment_variant=args.sentiment_variant,
    )

    dataset_path = default_processed_dataset_path(
        dataset_name=dataset_name,
        ticker=config.data.ticker,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. Build features before running the experiment."
        )

    dataset = load_processed_dataset(dataset_path)
    results = train_double_dqn(
        dataset=dataset,
        config=config,
        dataset_name=dataset_name,
        run_name=run_name,
    )

    summary = results["split_metrics"].copy()
    summary["reward_mode"] = config.environment.reward_mode
    summary["sentiment_variant"] = config.features.sentiment_variant

    summary_dir = Path("results/ablations")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}_{run_name}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Run: {run_name}")
    print(f"Reward mode: {config.environment.reward_mode}")
    print(f"Sentiment variant: {config.features.sentiment_variant}")
    print(f"Dataset: {dataset_name}")
    print(f"Best checkpoint: {results['best_checkpoint_path']}")
    print(f"Training log: {results['training_log_path']}")
    print(f"Metrics: {results['metrics_path']}")
    print(f"Summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
