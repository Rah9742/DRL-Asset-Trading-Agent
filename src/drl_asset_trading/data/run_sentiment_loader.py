"""Simple entry point for downloading and caching sentiment data."""

from __future__ import annotations

import argparse

import pandas as pd

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
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload raw sentiment data even if cached raw/interim artifacts exist.",
    )
    return parser.parse_args()


def _date_to_alpha_vantage_timestamp(date_value: str, end_of_day: bool) -> str:
    """Convert a YYYY-MM-DD date string into Alpha Vantage timestamp format."""
    suffix = "2359" if end_of_day else "0000"
    return f"{date_value.replace('-', '')}T{suffix}"


def build_sentiment_query(
    config: ExperimentConfig,
    ticker: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    topics: str | None = None,
    sort: str = "LATEST",
    limit: int = 1000,
) -> SentimentQuery:
    """Build a sentiment query using config defaults with optional overrides."""
    return SentimentQuery(
        ticker=ticker or config.data.ticker,
        time_from=time_from or _date_to_alpha_vantage_timestamp(config.data.start_date, end_of_day=False),
        time_to=time_to or _date_to_alpha_vantage_timestamp(config.data.end_date, end_of_day=True),
        topics=topics,
        sort=sort,
        limit=limit,
    )


def load_and_cache_sentiment_data(
    config: ExperimentConfig,
    ticker: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    topics: str | None = None,
    sort: str = "LATEST",
    limit: int = 1000,
    force_download: bool = False,
) -> dict[str, object]:
    """Load or rebuild sentiment artifacts, downloading only when required."""
    query = build_sentiment_query(
        config=config,
        ticker=ticker,
        time_from=time_from,
        time_to=time_to,
        topics=topics,
        sort=sort,
        limit=limit,
    )
    loader = SentimentDataLoader()
    paths = loader.default_paths(query)
    loaded_from_cache = False

    if paths.raw_json.exists() and not force_download:
        payload = loader.load_raw_json(paths.raw_json)
        loaded_from_cache = True
    else:
        payload = loader.fetch_all(query)
        loader.save_raw_json(payload, paths.raw_json)

    if paths.articles_csv.exists() and not force_download:
        articles = loader.load_articles_csv(paths.articles_csv)
    else:
        articles = loader.normalize_articles(payload, ticker=query.ticker)
        loader.save_articles_csv(articles, paths.articles_csv)

    if paths.daily_sentiment_csv.exists() and not force_download:
        daily_features = pd.read_csv(paths.daily_sentiment_csv)
    else:
        daily_features = loader.aggregate_daily_features(articles)
        loader.save_daily_features_csv(daily_features, paths.daily_sentiment_csv)

    return {
        "query": query,
        "paths": paths,
        "payload": payload,
        "articles": articles,
        "daily_features": daily_features,
        "loaded_from_cache": loaded_from_cache,
    }


def main() -> None:
    """Download, normalize, and aggregate sentiment data."""
    args = parse_args()
    load_env_file(args.env_file)
    config = ExperimentConfig.from_json(args.config)
    result = load_and_cache_sentiment_data(
        config=config,
        ticker=args.ticker,
        time_from=args.time_from,
        time_to=args.time_to,
        topics=args.topics,
        sort=args.sort,
        limit=args.limit,
        force_download=args.force_download,
    )
    query = result["query"]
    payload = result["payload"]
    articles = result["articles"]
    paths = result["paths"]

    print(f"Ticker: {query.ticker}")
    print(f"Articles retrieved: {len(articles)}")
    print(f"Batches fetched: {payload.get('pagination', {}).get('batch_count', 1)}")
    print(f"Raw JSON: {paths.raw_json}")
    print(f"Articles CSV: {paths.articles_csv}")
    print(f"Daily sentiment CSV: {paths.daily_sentiment_csv}")


if __name__ == "__main__":
    main()
