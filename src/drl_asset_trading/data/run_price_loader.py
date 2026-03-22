"""Simple entry point for downloading and caching market data."""

from __future__ import annotations

import argparse

from ..config import ExperimentConfig, load_env_file
from ..data.price_loader import MarketDataLoader


def load_and_cache_price_data(
    config: ExperimentConfig,
    force_download: bool = False,
) -> tuple[object, object]:
    """Load price data and ensure the cache is populated."""
    loader = MarketDataLoader(config.data)
    csv_path = loader.default_csv_path()

    if force_download and csv_path.exists():
        csv_path.unlink()

    data = loader.load()
    return data, csv_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data loader runner."""
    parser = argparse.ArgumentParser(description="Load and cache market data.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the local .env file.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore an existing cached CSV and download fresh data.",
    )
    return parser.parse_args()


def main() -> None:
    """Load data using the configured provider and print a short summary."""
    args = parse_args()
    load_env_file(args.env_file)
    config = ExperimentConfig.from_json(args.config)
    data, csv_path = load_and_cache_price_data(config=config, force_download=args.force_download)

    print(f"Loaded ticker: {config.data.ticker}")
    print(f"Provider: {config.data.provider}")
    print(f"CSV path: {csv_path}")
    print(f"Date range: {data.index.min().date()} -> {data.index.max().date()}")
    print(f"Shape: {data.shape}")
    print("Saved and ready for reuse.")


if __name__ == "__main__":
    main()
