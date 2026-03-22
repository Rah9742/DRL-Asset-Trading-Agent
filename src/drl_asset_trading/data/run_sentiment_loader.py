"""Simple entry point for downloading and caching sentiment data."""

from __future__ import annotations

import argparse

from ..config import ExperimentConfig, load_env_file
from .sentiment_loader import SentimentDataLoader, SentimentQuery


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sentiment loader runner."""
    parser = argparse.ArgumentParser(description="Download and store sentiment data.")
    parser.add_argument(
        "--config",
        default="configs/baseline_experiment.json",
        help="Path to the experiment config JSON.",
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol, for example SPY.")
    parser.add_argument("--time-from", default=None, help="Lower bound in YYYYMMDDTHHMM format.")
    parser.add_argument("--time-to", default=None, help="Upper bound in YYYYMMDDTHHMM format.")
    parser.add_argument("--topics", default=None, help="Comma-separated Alpha Vantage topics.")
    parser.add_argument("--sort", default="LATEST", help="LATEST, EARLIEST, or RELEVANCE.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum article count.")
    parser.add_argument("--env-file", default=".env", help="Path to the local .env file.")
    return parser.parse_args()


def _date_to_alpha_vantage_timestamp(date_value: str, end_of_day: bool) -> str:
    """Convert a YYYY-MM-DD date string into Alpha Vantage timestamp format."""
    suffix = "2359" if end_of_day else "0000"
    return f"{date_value.replace('-', '')}T{suffix}"


def main() -> None:
    """Download, normalize, and aggregate sentiment data."""
    args = parse_args()
    load_env_file(args.env_file)
    config = ExperimentConfig.from_json(args.config)

    query = SentimentQuery(
        ticker=args.ticker or config.data.ticker,
        time_from=args.time_from or _date_to_alpha_vantage_timestamp(config.data.start_date, end_of_day=False),
        time_to=args.time_to or _date_to_alpha_vantage_timestamp(config.data.end_date, end_of_day=True),
        topics=args.topics,
        sort=args.sort,
        limit=args.limit,
    )
    loader = SentimentDataLoader()
    paths = loader.default_paths(query)

    payload = loader.fetch_all(query)
    articles = loader.normalize_articles(payload, ticker=query.ticker)
    daily_features = loader.aggregate_daily_features(articles)

    loader.save_raw_json(payload, paths.raw_json)
    loader.save_articles_csv(articles, paths.articles_csv)
    loader.save_daily_features_csv(daily_features, paths.daily_sentiment_csv)

    print(f"Ticker: {query.ticker}")
    print(f"Articles retrieved: {len(articles)}")
    print(f"Batches fetched: {payload.get('pagination', {}).get('batch_count', 1)}")
    print(f"Raw JSON: {paths.raw_json}")
    print(f"Articles CSV: {paths.articles_csv}")
    print(f"Daily sentiment CSV: {paths.daily_sentiment_csv}")


if __name__ == "__main__":
    main()
