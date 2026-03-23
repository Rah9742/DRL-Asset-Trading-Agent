"""Build processed price-only and price-plus-sentiment datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..data.price_loader import MarketDataLoader
from .engineering import FeatureBuilder, load_sentiment_daily_csv


def build_and_save_feature_datasets(
    config: ExperimentConfig,
    skip_sentiment: bool = False,
) -> dict[str, Path]:
    """Build and save processed datasets for all available state variants."""
    price_loader = MarketDataLoader(config.data)
    price_path = price_loader.default_csv_path()
    if not price_path.exists():
        raise FileNotFoundError(
            f"Price CSV not found at {price_path}. Run the price loader before building features."
        )

    price_data = price_loader.load_csv(price_path)
    feature_builder = FeatureBuilder(config.features)
    output_paths = feature_builder.default_processed_paths(
        ticker=config.data.ticker,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
    )

    sentiment_path = Path("data/interim/sentiment/daily") / f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}.csv"
    sentiment_daily = None
    if not skip_sentiment and sentiment_path.exists():
        sentiment_daily = load_sentiment_daily_csv(sentiment_path)

    datasets = feature_builder.build_feature_sets(price_data=price_data, sentiment_daily=sentiment_daily)
    saved_paths: dict[str, Path] = {}
    for dataset_name, dataset in datasets.items():
        saved_paths[dataset_name] = feature_builder.save_dataset(dataset, output_paths[dataset_name])
    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the feature builder."""
    parser = argparse.ArgumentParser(description="Build processed feature datasets.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Build only the price dataset even if sentiment data exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save processed feature datasets."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    saved_paths = build_and_save_feature_datasets(config=config, skip_sentiment=args.skip_sentiment)
    for dataset_name, dataset_path in saved_paths.items():
        print(f"{dataset_name} dataset: {dataset_path}")

    sentiment_path = Path("data/interim/sentiment/daily") / f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}.csv"
    if "price_sentiment_decay" in saved_paths:
        print(f"Sentiment source: {sentiment_path}")
    else:
        print("Sentiment datasets not built. No interim sentiment daily CSV was found.")


if __name__ == "__main__":
    main()
