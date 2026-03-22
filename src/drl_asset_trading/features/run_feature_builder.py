"""Build processed baseline and augmented feature datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ExperimentConfig
from ..data.price_loader import MarketDataLoader
from .engineering import FeatureBuilder, load_sentiment_daily_csv


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
        help="Build only the baseline dataset even if sentiment data exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save processed baseline and augmented datasets."""
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)

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
    if not args.skip_sentiment and sentiment_path.exists():
        sentiment_daily = load_sentiment_daily_csv(sentiment_path)

    datasets = feature_builder.build_feature_sets(price_data=price_data, sentiment_daily=sentiment_daily)

    baseline_path = feature_builder.save_dataset(datasets["baseline"], output_paths["baseline"])
    print(f"Baseline dataset: {baseline_path}")
    print(f"Baseline rows: {len(datasets['baseline'])}")

    if "augmented" in datasets:
        augmented_path = feature_builder.save_dataset(datasets["augmented"], output_paths["augmented"])
        print(f"Augmented dataset: {augmented_path}")
        print(f"Augmented rows: {len(datasets['augmented'])}")
        print(f"Sentiment source: {sentiment_path}")
    else:
        print("Augmented dataset not built. No interim sentiment daily CSV was found.")


if __name__ == "__main__":
    main()
