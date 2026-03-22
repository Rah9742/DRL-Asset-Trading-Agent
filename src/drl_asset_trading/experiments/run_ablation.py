"""Run the four-way Double DQN ablation study."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..agents.training import train_double_dqn
from ..config import ExperimentConfig
from ..evaluation.benchmarks import default_processed_dataset_path, load_processed_dataset
from ..features import resolve_processed_dataset_name

ABLATION_VARIANTS = {
    "baseline": {"state_mode": "price_only", "reward_mode": "profit"},
    "reward_only": {"state_mode": "price_only", "reward_mode": "differential_sharpe"},
    "state_only": {"state_mode": "price_sentiment", "reward_mode": "profit"},
    "both": {"state_mode": "price_sentiment", "reward_mode": "differential_sharpe"},
}


def configure_variant(
    config: ExperimentConfig,
    variant_name: str,
    sentiment_imputation_mode: str | None = None,
) -> tuple[ExperimentConfig, str, str]:
    """Apply ablation settings to a config and resolve the processed dataset name."""
    variant = ABLATION_VARIANTS[variant_name]
    config.features.state_mode = variant["state_mode"]
    config.environment.reward_mode = variant["reward_mode"]
    if config.features.state_mode == "price_only":
        config.features.sentiment_imputation_mode = "none"
    elif sentiment_imputation_mode is not None:
        config.features.sentiment_imputation_mode = sentiment_imputation_mode

    dataset_name = resolve_processed_dataset_name(
        state_mode=config.features.state_mode,
        sentiment_imputation_mode=config.features.sentiment_imputation_mode,
    )
    imputation_label = config.features.sentiment_imputation_mode if config.features.state_mode == "price_sentiment" else "none"
    run_name = f"{variant_name}_{imputation_label}"
    return config, dataset_name, run_name


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation runner."""
    parser = argparse.ArgumentParser(description="Run a Double DQN ablation variant.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(ABLATION_VARIANTS),
        required=True,
        help="Ablation variant to run.",
    )
    parser.add_argument(
        "--sentiment-imputation-mode",
        choices=["zero", "decay"],
        default=None,
        help="Sentiment imputation mode used when state_mode=price_sentiment.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a single ablation variant using the common Double DQN pipeline."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    config, dataset_name, run_name = configure_variant(
        config=config,
        variant_name=args.variant,
        sentiment_imputation_mode=args.sentiment_imputation_mode,
    )
    dataset_path = default_processed_dataset_path(
        dataset_name=dataset_name,
        ticker=config.data.ticker,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {dataset_path}. Build features before running the ablation."
        )

    dataset = load_processed_dataset(dataset_path)
    results = train_double_dqn(
        dataset=dataset,
        config=config,
        dataset_name=dataset_name,
        run_name=run_name,
    )

    summary = results["split_metrics"].copy()
    summary["variant"] = args.variant
    summary["dataset_name"] = dataset_name
    summary["state_mode"] = config.features.state_mode
    summary["reward_mode"] = config.environment.reward_mode
    summary["sentiment_imputation_mode"] = config.features.sentiment_imputation_mode

    summary_dir = Path("results/ablations")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}_{run_name}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Variant: {args.variant}")
    print(f"State mode: {config.features.state_mode}")
    print(f"Reward mode: {config.environment.reward_mode}")
    print(f"Sentiment imputation mode: {config.features.sentiment_imputation_mode}")
    print(f"Dataset: {dataset_name}")
    print(f"Best checkpoint: {results['best_checkpoint_path']}")
    print(f"Training log: {results['training_log_path']}")
    print(f"Metrics: {results['metrics_path']}")
    print(f"Ablation summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
