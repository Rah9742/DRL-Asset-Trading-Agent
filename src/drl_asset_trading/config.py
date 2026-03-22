"""Configuration helpers for reproducible trading experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_env_file(env_path: str | Path = ".env") -> dict[str, str]:
    """Load key-value pairs from a local .env file without extra dependencies."""
    path = Path(env_path)
    if not path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip()
        os.environ.setdefault(key.strip(), value.strip())
    return loaded


@dataclass(slots=True)
class DataConfig:
    provider: str = "yfinance"
    ticker: str = "SPY"
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    interval: str = "1d"


@dataclass(slots=True)
class FeatureConfig:
    include_returns: bool = True
    include_log_returns: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_rsi: bool = True
    lookback_window: int = 14
    sentiment_lag_days: int = 1
    sentiment_fill_value: float = 0.0


@dataclass(slots=True)
class EnvironmentConfig:
    initial_cash: float = 10_000.0
    transaction_cost: float = 0.001
    reward_mode: str = "profit"
    annualization_factor: int = 252


@dataclass(slots=True)
class SplitConfig:
    train_end: str = "2020-12-31"
    validation_end: str = "2022-12-31"
    test_end: str = "2024-12-31"


@dataclass(slots=True)
class ExperimentOptions:
    random_seed: int = 42


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    features: FeatureConfig
    environment: EnvironmentConfig
    splits: SplitConfig
    experiment: ExperimentOptions

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        """Build a typed experiment config from a dictionary payload."""
        return cls(
            data=DataConfig(**payload.get("data", {})),
            features=FeatureConfig(**payload.get("features", {})),
            environment=EnvironmentConfig(**payload.get("environment", {})),
            splits=SplitConfig(**payload.get("splits", {})),
            experiment=ExperimentOptions(**payload.get("experiment", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment settings from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
