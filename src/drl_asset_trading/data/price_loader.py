"""Price data loading, caching, and split helpers."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

from ..config import DataConfig, SplitConfig


EXPECTED_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


@dataclass(slots=True)
class MarketDataLoader:
    """Load single-asset OHLCV data from supported providers and cache it locally."""

    config: DataConfig

    def load(self) -> pd.DataFrame:
        """Load cached data if available, otherwise download and cache it."""
        csv_path = self.default_csv_path()
        if csv_path.exists():
            return self.load_csv(csv_path)

        provider = self.config.provider.lower()
        if provider == "yfinance":
            data = self._load_from_yfinance()
        elif provider == "alphavantage":
            data = self._load_from_alpha_vantage()
        else:
            raise ValueError(
                f"Unsupported provider '{self.config.provider}'. "
                "Expected one of: 'yfinance', 'alphavantage'."
            )

        self.save_csv(data, csv_path)
        return data

    def default_csv_path(self) -> Path:
        """Return the default cache path for the configured dataset."""
        filename = (
            f"{self.config.ticker}_{self.config.start_date}_{self.config.end_date}.csv"
        )
        return Path("data/raw/price") / filename

    def save_csv(self, data: pd.DataFrame, path: str | Path) -> Path:
        """Save a normalized market data frame to CSV."""
        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_path, index_label="Date")
        return csv_path

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load a cached CSV and normalize its structure."""
        frame = pd.read_csv(path, index_col="Date", parse_dates=True)
        return _finalize_frame(frame, self.config.ticker)

    def _load_from_yfinance(self) -> pd.DataFrame:
        data = yf.download(
            tickers=self.config.ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            interval=self.config.interval,
            auto_adjust=False,
            progress=False,
        )
        return _normalize_downloaded_frame(data, self.config.ticker)

    def _load_from_alpha_vantage(self) -> pd.DataFrame:
        if self.config.interval != "1d":
            raise ValueError("Alpha Vantage support is limited to daily data in this scaffold.")

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is not set. Add it to your .env file.")

        query = urlencode(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": self.config.ticker,
                "outputsize": "full",
                "datatype": "csv",
                "apikey": api_key,
            }
        )
        url = f"https://www.alphavantage.co/query?{query}"

        try:
            with urlopen(url, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise ValueError(f"Failed to download Alpha Vantage data: {exc}") from exc

        if not payload.strip():
            raise ValueError(f"No data returned for ticker '{self.config.ticker}' from Alpha Vantage.")

        frame = pd.read_csv(StringIO(payload))
        if "timestamp" not in frame.columns:
            snippet = _safe_error_snippet(payload)
            raise ValueError(f"Unexpected Alpha Vantage response for '{self.config.ticker}': {snippet}")

        normalized = frame.rename(
            columns={
                "timestamp": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adjusted_close": "Adj Close",
                "volume": "Volume",
            }
        )
        normalized["Date"] = pd.to_datetime(normalized["Date"])
        normalized = normalized.set_index("Date").sort_index()
        normalized = normalized.loc[self.config.start_date : self.config.end_date].copy()
        return _finalize_frame(normalized, self.config.ticker)


def split_by_dates(data: pd.DataFrame, split_config: SplitConfig) -> dict[str, pd.DataFrame]:
    """Split a time-indexed frame into train, validation, and test windows."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a DatetimeIndex for date-based splits.")

    train = data.loc[: split_config.train_end].copy()
    validation = data.loc[split_config.train_end : split_config.validation_end].iloc[1:].copy()
    test = data.loc[split_config.validation_end : split_config.test_end].iloc[1:].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("One or more dataset splits are empty. Check the configured date boundaries.")

    return {"train": train, "validation": validation, "test": test}


def _normalize_downloaded_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize provider output into a consistent single-asset OHLCV frame."""
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return _finalize_frame(frame, ticker)


def _finalize_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate, sort, and annotate a market data frame."""
    missing = [column for column in EXPECTED_PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}")

    normalized = frame.loc[:, EXPECTED_PRICE_COLUMNS].copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    normalized.index.name = "Date"
    normalized["Ticker"] = ticker
    return normalized


def _safe_error_snippet(payload: str) -> str:
    """Extract a short diagnostic string from a provider response."""
    stripped = payload.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed)[:200]
        except json.JSONDecodeError:
            return stripped[:200]

    reader = csv.reader(StringIO(stripped))
    first_row = next(reader, [])
    return ",".join(first_row)[:200]
