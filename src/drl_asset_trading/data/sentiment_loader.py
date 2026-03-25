"""Sentiment data loading, normalization, and aggregation helpers."""

from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi
import pandas as pd


@dataclass(slots=True)
class SentimentQuery:
    """Query parameters for the Alpha Vantage news sentiment API."""

    ticker: str
    time_from: str | None = None
    time_to: str | None = None
    topics: str | None = None
    sort: str = "LATEST"
    limit: int = 1000


@dataclass(slots=True)
class SentimentPaths:
    """Output paths for raw, normalized, and aggregated sentiment data."""

    raw_json: Path
    articles_csv: Path
    daily_sentiment_csv: Path


class SentimentDataLoader:
    """Download and store single-asset news sentiment data from Alpha Vantage."""

    base_url = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is not set. Add it to your .env file.")
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def fetch(self, query: SentimentQuery) -> dict:
        """Fetch raw news sentiment data from Alpha Vantage."""
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": query.ticker,
            "sort": query.sort,
            "limit": str(query.limit),
            "apikey": self.api_key,
        }
        if query.time_from:
            params["time_from"] = query.time_from
        if query.time_to:
            params["time_to"] = query.time_to
        if query.topics:
            params["topics"] = query.topics

        url = f"{self.base_url}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=30, context=self.ssl_context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to retrieve sentiment data: {exc}") from exc

        if "feed" not in payload:
            diagnostic = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or payload
            raise ValueError(f"Unexpected Alpha Vantage response: {diagnostic}")
        return payload

    def fetch_all(self, query: SentimentQuery) -> dict:
        """Fetch all available articles for a query by paging over time windows."""
        if not query.time_from or not query.time_to:
            return self.fetch(query)

        batches: list[dict] = []
        combined_feed: list[dict] = []
        current_from = query.time_from
        requested_limit = min(query.limit, 1000)
        batch_index = 0
        first_payload: dict | None = None

        while True:
            batch_index += 1
            batch_query = SentimentQuery(
                ticker=query.ticker,
                time_from=current_from,
                time_to=query.time_to,
                topics=query.topics,
                sort="EARLIEST",
                limit=requested_limit,
            )
            payload = self.fetch(batch_query)
            if first_payload is None:
                first_payload = payload
            feed = payload.get("feed", [])
            batches.append(
                {
                    "batch_index": batch_index,
                    "time_from": current_from,
                    "time_to": query.time_to,
                    "articles_retrieved": len(feed),
                }
            )

            if not feed:
                break

            combined_feed.extend(feed)
            if len(feed) < requested_limit:
                break

            last_timestamp = feed[-1].get("time_published")
            if not last_timestamp:
                break

            next_from = _increment_alpha_vantage_timestamp(last_timestamp)
            if next_from > query.time_to:
                break
            current_from = next_from

        deduplicated_feed = _deduplicate_articles(combined_feed)
        response = {
            "items": str(len(deduplicated_feed)),
            "feed": deduplicated_feed,
            "pagination": {
                "requested_time_from": query.time_from,
                "requested_time_to": query.time_to,
                "batch_limit": requested_limit,
                "batch_count": len(batches),
                "batches": batches,
            },
        }

        if first_payload is not None:
            for key in ("sentiment_score_definition", "relevance_score_definition"):
                if key in first_payload:
                    response[key] = first_payload[key]

        return response

    @staticmethod
    def default_paths(query: SentimentQuery) -> SentimentPaths:
        """Build default output paths for a given sentiment query."""
        start_label = _timestamp_to_date_label(query.time_from) if query.time_from else "start"
        end_label = _timestamp_to_date_label(query.time_to) if query.time_to else "latest"
        stem = f"{query.ticker}_{start_label}_{end_label}"
        return SentimentPaths(
            raw_json=Path("data/raw/sentiment") / f"{stem}.json",
            articles_csv=Path("data/interim/sentiment/articles") / f"{stem}.csv",
            daily_sentiment_csv=Path("data/interim/sentiment/daily") / f"{stem}.csv",
        )

    def save_raw_json(self, payload: dict, path: str | Path) -> Path:
        """Persist the raw API response for reproducibility."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def load_raw_json(self, path: str | Path) -> dict:
        """Load a cached raw API response from disk."""
        input_path = Path(path)
        return json.loads(input_path.read_text(encoding="utf-8"))

    def normalize_articles(self, payload: dict, ticker: str) -> pd.DataFrame:
        """Flatten raw article data into a tabular article-level dataframe."""
        rows: list[dict[str, object]] = []
        for article in payload.get("feed", []):
            ticker_match = self._extract_ticker_sentiment(article.get("ticker_sentiment", []), ticker)
            topics = article.get("topics", [])
            authors = article.get("authors", [])

            rows.append(
                {
                    "ticker": ticker,
                    "time_published": pd.to_datetime(article.get("time_published"), format="%Y%m%dT%H%M%S", utc=True),
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "source": article.get("source"),
                    "source_domain": article.get("source_domain"),
                    "summary": article.get("summary"),
                    "authors": "|".join(authors),
                    "topic_names": "|".join(topic.get("topic", "") for topic in topics),
                    "topic_relevance_mean": _mean_or_none(topic.get("relevance_score") for topic in topics),
                    "overall_sentiment_score": _to_float(article.get("overall_sentiment_score")),
                    "overall_sentiment_label": article.get("overall_sentiment_label"),
                    "ticker_relevance_score": ticker_match.get("relevance_score"),
                    "ticker_sentiment_score": ticker_match.get("ticker_sentiment_score"),
                    "ticker_sentiment_label": ticker_match.get("ticker_sentiment_label"),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "time_published",
                    "title",
                    "url",
                    "source",
                    "source_domain",
                    "summary",
                    "authors",
                    "topic_names",
                    "topic_relevance_mean",
                    "overall_sentiment_score",
                    "overall_sentiment_label",
                    "ticker_relevance_score",
                    "ticker_sentiment_score",
                    "ticker_sentiment_label",
                ]
            )

        frame = pd.DataFrame(rows).sort_values("time_published").reset_index(drop=True)
        return frame

    def save_articles_csv(self, articles: pd.DataFrame, path: str | Path) -> Path:
        """Save normalized article-level data."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        articles.to_csv(output_path, index=False)
        return output_path

    def load_articles_csv(self, path: str | Path) -> pd.DataFrame:
        """Load cached article-level sentiment data from disk."""
        return pd.read_csv(path, parse_dates=["time_published"])

    def aggregate_daily_features(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Aggregate article-level sentiment into daily state features."""
        if articles.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "news_count",
                    "mean_overall_sentiment",
                    "mean_ticker_sentiment",
                    "mean_ticker_relevance",
                    "weighted_ticker_sentiment",
                    "sentiment_std",
                ]
            )

        frame = articles.copy()
        frame["date"] = pd.to_datetime(frame["time_published"]).dt.tz_convert("UTC").dt.date
        frame["weighted_sentiment_component"] = (
            frame["ticker_sentiment_score"].fillna(frame["overall_sentiment_score"]).fillna(0.0)
            * frame["ticker_relevance_score"].fillna(0.0)
        )

        grouped = frame.groupby("date", as_index=False).agg(
            news_count=("title", "count"),
            mean_overall_sentiment=("overall_sentiment_score", "mean"),
            mean_ticker_sentiment=("ticker_sentiment_score", "mean"),
            mean_ticker_relevance=("ticker_relevance_score", "mean"),
            sentiment_std=("ticker_sentiment_score", "std"),
            weighted_sentiment_sum=("weighted_sentiment_component", "sum"),
            relevance_sum=("ticker_relevance_score", "sum"),
        )
        grouped["weighted_ticker_sentiment"] = grouped["weighted_sentiment_sum"] / grouped["relevance_sum"].replace(0.0, pd.NA)
        grouped["sentiment_std"] = grouped["sentiment_std"].fillna(0.0)
        grouped = grouped.drop(columns=["weighted_sentiment_sum", "relevance_sum"])
        return grouped

    def save_daily_features_csv(self, daily_features: pd.DataFrame, path: str | Path) -> Path:
        """Save aggregated daily sentiment data to the interim layer."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily_features.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def _extract_ticker_sentiment(entries: list[dict], ticker: str) -> dict[str, object]:
        """Return the sentiment entry matching the requested ticker, if present."""
        for entry in entries:
            if entry.get("ticker") == ticker:
                return {
                    "relevance_score": _to_float(entry.get("relevance_score")),
                    "ticker_sentiment_score": _to_float(entry.get("ticker_sentiment_score")),
                    "ticker_sentiment_label": entry.get("ticker_sentiment_label"),
                }
        return {
            "relevance_score": None,
            "ticker_sentiment_score": None,
            "ticker_sentiment_label": None,
        }


def _to_float(value: object) -> float | None:
    """Convert API string values to floats when possible."""
    if value in (None, ""):
        return None
    return float(value)


def _mean_or_none(values) -> float | None:
    """Return the mean of non-null numeric values, or None."""
    numeric_values = [float(value) for value in values if value not in (None, "")]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _timestamp_to_date_label(timestamp: str) -> str:
    """Convert an Alpha Vantage timestamp into a YYYY-MM-DD filename label."""
    parsed = pd.to_datetime(timestamp, format="%Y%m%dT%H%M")
    return parsed.strftime("%Y-%m-%d")


def _increment_alpha_vantage_timestamp(timestamp: str) -> str:
    """Advance an Alpha Vantage article timestamp by one second."""
    parsed = pd.to_datetime(timestamp, format="%Y%m%dT%H%M%S")
    incremented = parsed + pd.Timedelta(seconds=1)
    return incremented.strftime("%Y%m%dT%H%M")


def _deduplicate_articles(feed: list[dict]) -> list[dict]:
    """Deduplicate articles using URL first, then a fallback composite key."""
    deduplicated: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for article in sorted(feed, key=lambda item: item.get("time_published", "")):
        url = article.get("url") or ""
        title = article.get("title") or ""
        published = article.get("time_published") or ""
        key = ("url", url, "") if url else ("article", published, title)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(article)

    return deduplicated
