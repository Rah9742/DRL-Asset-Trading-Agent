"""Feature engineering for baseline and extended state representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FeatureConfig

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
SENTIMENT_COLUMNS = [
    "news_count",
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
        """Create a baseline set of price-derived and technical features."""
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
        """Create baseline and optionally sentiment-augmented feature datasets."""
        baseline = self.transform(price_data)
        datasets = {"baseline": baseline}

        if sentiment_daily is not None and not sentiment_daily.empty:
            augmented = self.merge_sentiment_features(baseline, sentiment_daily)
            datasets["augmented"] = augmented

        return datasets

    def merge_sentiment_features(self, price_features: pd.DataFrame, sentiment_daily: pd.DataFrame) -> pd.DataFrame:
        """Merge lagged daily sentiment data onto price-derived features."""
        if "date" not in sentiment_daily.columns:
            raise ValueError("Sentiment daily data must contain a 'date' column.")

        sentiment = sentiment_daily.copy()
        sentiment["date"] = pd.to_datetime(sentiment["date"])
        sentiment = sentiment.sort_values("date")
        sentiment = sentiment.set_index("date")
        sentiment.index = sentiment.index + pd.Timedelta(days=self.config.sentiment_lag_days)
        sentiment = sentiment.reindex(columns=SENTIMENT_COLUMNS)
        sentiment = sentiment.fillna(self.config.sentiment_fill_value)

        augmented = price_features.copy()
        augmented.index = pd.to_datetime(augmented.index).normalize()
        augmented = augmented.join(sentiment, how="left")
        augmented[SENTIMENT_COLUMNS] = augmented[SENTIMENT_COLUMNS].fillna(self.config.sentiment_fill_value)
        return augmented

    @staticmethod
    def default_processed_paths(ticker: str, start_date: str, end_date: str) -> dict[str, Path]:
        """Return default output paths for processed baseline and augmented datasets."""
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        return {
            "baseline": Path("data/processed/baseline") / filename,
            "augmented": Path("data/processed/augmented") / filename,
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
