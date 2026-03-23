"""Single entry point for the full coursework pipeline."""

from __future__ import annotations

import argparse

from .config import ExperimentConfig, load_env_file
from .data.price_loader import MarketDataLoader
from .data.run_price_loader import load_and_cache_price_data
from .data.run_sentiment_loader import build_sentiment_query, load_and_cache_sentiment_data
from .data.sentiment_loader import SentimentDataLoader
from .experiments.run_full_comparison import run_full_comparison
from .features.run_feature_builder import build_and_save_feature_datasets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the orchestration entry point."""
    parser = argparse.ArgumentParser(description="Run the full DRL asset-trading pipeline.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Override the ticker defined in the config file.",
    )
    parser.add_argument(
        "--sentiment-imputation-mode",
        choices=["zero", "decay"],
        default="decay",
        help="Sentiment imputation mode for price_sentiment variants.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the local .env file.",
    )
    parser.add_argument(
        "--force-price-download",
        action="store_true",
        help="Redownload the price CSV even if it already exists.",
    )
    parser.add_argument(
        "--sentiment-topics",
        default=None,
        help="Optional comma-separated Alpha Vantage topic filter for sentiment ingestion.",
    )
    parser.add_argument(
        "--sentiment-sort",
        default="LATEST",
        help="Alpha Vantage sentiment sort order.",
    )
    parser.add_argument(
        "--sentiment-limit",
        type=int,
        default=1000,
        help="Alpha Vantage sentiment batch limit.",
    )
    parser.add_argument(
        "--force-sentiment-download",
        action="store_true",
        help="Redownload sentiment artifacts even if the cached raw/interim files already exist.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list overriding experiment.seed_values.",
    )
    return parser.parse_args()


def main() -> None:
    """Run price ingestion, sentiment ingestion, feature building, and full comparison."""
    args = parse_args()
    load_env_file(args.env_file)
    config = ExperimentConfig.from_json(args.config)

    if args.ticker:
        config.data.ticker = args.ticker

    print(f"Running pipeline for ticker {config.data.ticker}", flush=True)

    price_path = MarketDataLoader(config.data).default_csv_path()
    if price_path.exists() and not args.force_price_download:
        print("Step 1/4: using cached price data", flush=True)
    else:
        print("Step 1/4: loading price data", flush=True)
    _, price_path = load_and_cache_price_data(
        config=config,
        force_download=args.force_price_download,
    )
    print(f"Price CSV ready: {price_path}", flush=True)

    sentiment_query = build_sentiment_query(
        config=config,
        ticker=config.data.ticker,
        topics=args.sentiment_topics,
        sort=args.sentiment_sort,
        limit=args.sentiment_limit,
    )
    sentiment_paths = SentimentDataLoader.default_paths(sentiment_query)
    sentiment_cache_ready = (
        sentiment_paths.raw_json.exists()
        and sentiment_paths.articles_csv.exists()
        and sentiment_paths.daily_sentiment_csv.exists()
    )

    if sentiment_cache_ready and not args.force_sentiment_download:
        print("Step 2/4: using cached sentiment data", flush=True)
        print(f"Sentiment daily CSV ready: {sentiment_paths.daily_sentiment_csv}", flush=True)
    else:
        print("Step 2/4: loading sentiment data", flush=True)
        sentiment_result = load_and_cache_sentiment_data(
            config=config,
            ticker=config.data.ticker,
            topics=args.sentiment_topics,
            sort=args.sentiment_sort,
            limit=args.sentiment_limit,
        )
        print(f"Sentiment daily CSV ready: {sentiment_result['paths'].daily_sentiment_csv}", flush=True)

    print("Step 3/4: building processed feature datasets", flush=True)
    dataset_paths = build_and_save_feature_datasets(config=config, skip_sentiment=False)
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"{dataset_name} dataset ready: {dataset_path}", flush=True)

    print("Step 4/4: running full comparison", flush=True)
    results = run_full_comparison(
        config=config,
        sentiment_imputation_mode=args.sentiment_imputation_mode,
        seeds=[int(chunk.strip()) for chunk in args.seeds.split(",") if chunk.strip()] if args.seeds else None,
    )
    print(f"Comparison table: {results['comparison_path']}", flush=True)
    print(f"Equity plot: {results['equity_plot_path']}", flush=True)
    print(f"Drawdown plot: {results['drawdown_plot_path']}", flush=True)


if __name__ == "__main__":
    main()
