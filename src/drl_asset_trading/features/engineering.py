"""Feature engineering for price-only and price-plus-sentiment state datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FeatureConfig, normalize_sentiment_variant

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
SENTIMENT_COLUMNS = [
    "news_count",
    "mean_overall_sentiment",
    "mean_ticker_sentiment",
    "mean_ticker_relevance",
    "weighted_ticker_sentiment",
    "sentiment_std",
]
DECAY_SENTIMENT_COLUMNS = [
    "mean_overall_sentiment",
    "mean_ticker_sentiment",
    "mean_ticker_relevance",
    "weighted_ticker_sentiment",
    "sentiment_std",
]


class FeatureBuilder:
    """Build feature matrices from OHLCV market data."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create the core price-derived and technical feature set."""
        required_columns = {"Close", "Volume"}
        missing = required_columns.difference(data.columns)
        if missing:
            raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

        frame = data.copy()
        close = frame["Close"]
        lookback = self.config.lookback_window

        if self.config.include_returns:
            frame["return_1"] = close.pct_change()
        if self.config.include_log_returns:
            frame["log_return_1"] = np.log(close / close.shift(1))
        if self.config.include_momentum:
            frame[f"momentum_{lookback}"] = close / close.shift(lookback) - 1.0
        if self.config.include_volatility:
            frame[f"volatility_{lookback}"] = close.pct_change().rolling(lookback).std()
        if self.config.include_rsi:
            frame[f"rsi_{lookback}"] = self._rsi(close, lookback)

        frame["volume_change_1"] = frame["Volume"].pct_change()
        return frame.dropna().copy()

    def build_feature_sets(
        self,
        price_data: pd.DataFrame,
        sentiment_daily: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Create price-only and optional sentiment-augmented feature datasets."""
        price = self.transform(price_data)
        datasets = {"price": price}

        if sentiment_daily is not None and not sentiment_daily.empty:
            datasets["price_sentiment_sparse"] = self.merge_sentiment_features(
                price_features=price,
                sentiment_daily=sentiment_daily,
                imputation_mode="sparse",
            )
            datasets["price_sentiment_decay"] = self.merge_sentiment_features(
                price_features=price,
                sentiment_daily=sentiment_daily,
                imputation_mode="decay",
            )

        return datasets

    def merge_sentiment_features(
        self,
        price_features: pd.DataFrame,
        sentiment_daily: pd.DataFrame,
        imputation_mode: str | None = None,
    ) -> pd.DataFrame:
        """Merge lagged daily sentiment data onto price-derived features.

        Sparse mode:
        Missing sentiment values are replaced by `sentiment_fill_value`, while `news_count`
        remains available so the model can distinguish no-news days from low-sentiment days.

        Decay mode:
        The latest available sentiment observation is carried forward with exponential decay
        based on `days_since_last_news`. This creates an enriched state that reflects stale
        sentiment fading over time rather than disappearing immediately.
        """
        if "date" not in sentiment_daily.columns:
            raise ValueError("Sentiment daily data must contain a 'date' column.")
        mode = imputation_mode or self.config.sentiment_imputation_mode
        if mode == "zero":
            mode = "sparse"
        if mode not in {"sparse", "decay"}:
            raise ValueError(f"Unsupported sentiment imputation mode: {mode}")

        sentiment = sentiment_daily.copy()
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        sentiment = sentiment.sort_values("date")
        sentiment = sentiment.set_index("date")
        sentiment.index = sentiment.index + pd.Timedelta(days=self.config.sentiment_lag_days)
        sentiment = sentiment.reindex(columns=SENTIMENT_COLUMNS)

        augmented = price_features.copy()
        augmented.index = pd.to_datetime(augmented.index).normalize()
        augmented = augmented.join(sentiment, how="left")
        augmented["news_count"] = augmented["news_count"].fillna(0.0)

        if mode == "sparse":
            augmented[SENTIMENT_COLUMNS] = augmented[SENTIMENT_COLUMNS].fillna(self.config.sentiment_fill_value)
            return augmented

        return self._apply_decay_imputation(augmented)

    def _apply_decay_imputation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Apply exponential decay to carry sentiment forward across no-news days."""
        enriched = dataset.copy()
        sentiment_observed = enriched["news_count"] > 0

        last_news_date = pd.Series(enriched.index.where(sentiment_observed), index=enriched.index).ffill()
        days_since_last_news = (
            pd.Series(enriched.index, index=enriched.index) - pd.to_datetime(last_news_date)
        ).dt.days.fillna(0.0)
        enriched["days_since_last_news"] = days_since_last_news.astype(float)

        decay_factor = np.exp(-self.config.sentiment_decay_rate * enriched["days_since_last_news"])
        for column in DECAY_SENTIMENT_COLUMNS:
            forward_filled = enriched[column].ffill().fillna(self.config.sentiment_fill_value)
            enriched[column] = forward_filled * decay_factor

        enriched["news_count"] = enriched["news_count"].fillna(0.0)
        return enriched

    @staticmethod
    def default_processed_paths(ticker: str, start_date: str, end_date: str) -> dict[str, Path]:
        """Return default output paths for processed dataset variants."""
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        return {
            "price": Path("data/processed/price") / filename,
            "price_sentiment_sparse": Path("data/processed/price_sentiment_sparse") / filename,
            "price_sentiment_decay": Path("data/processed/price_sentiment_decay") / filename,
        }

    @staticmethod
    def save_dataset(dataset: pd.DataFrame, path: str | Path) -> Path:
        """Save a processed dataset to CSV."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index_label="Date")
        return output_path

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        """Compute a simple RSI indicator."""
        delta = series.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)

        avg_gain = gains.rolling(window=window, min_periods=window).mean()
        avg_loss = losses.rolling(window=window, min_periods=window).mean()
        relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        return rsi.fillna(50.0)


def load_sentiment_daily_csv(path: str | Path) -> pd.DataFrame:
    """Load an interim daily sentiment CSV."""
    return pd.read_csv(path)


def resolve_processed_dataset_name(sentiment_variant: str) -> str:
    """Map a sentiment variant to a processed dataset name."""
    sentiment_variant = normalize_sentiment_variant(sentiment_variant)
    if sentiment_variant == "none":
        return "price"
    if sentiment_variant == "sparse":
        return "price_sentiment_sparse"
    if sentiment_variant == "decay":
        return "price_sentiment_decay"
    raise ValueError(f"Unsupported sentiment variant: {sentiment_variant}")
