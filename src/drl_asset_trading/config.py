"""Configuration helpers for reproducible trading experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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
    sentiment_variant: str = "none"
    state_mode: str = "price_only"
    include_returns: bool = True
    include_log_returns: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_rsi: bool = True
    lookback_window: int = 14
    sentiment_lag_days: int = 1
    sentiment_fill_value: float = 0.0
    sentiment_imputation_mode: str = "zero"
    sentiment_decay_rate: float = 0.25


@dataclass(slots=True)
class EnvironmentConfig:
    initial_cash: float = 10_000.0
    transaction_cost: float = 0.001
    reward_mode: str = "profit"
    annualization_factor: int = 252
    differential_sharpe_eta: float = 0.005
    differential_sharpe_eta_candidates: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
    differential_sharpe_epsilon: float = 1e-8
    differential_sharpe_warmup_steps: int = 20
    differential_sharpe_min_variance: float = 1e-6


@dataclass(slots=True)
class SplitConfig:
    train_end: str = "2020-12-31"
    validation_end: str = "2022-12-31"
    test_end: str = "2024-12-31"


@dataclass(slots=True)
class ExperimentOptions:
    random_seed: int = 42
    seed_values: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])


@dataclass(slots=True)
class RLConfig:
    algorithm: str = "double_dqn"
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 10_000
    target_update_frequency: int = 100
    training_episodes: int = 50
    warmup_steps: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    validation_metric: str = "sharpe_ratio"
    checkpoint_dir: str = "checkpoints/double_dqn"
    results_dir: str = "results/double_dqn"
    gradient_clip_norm: float = 1.0
    device: str = "cpu"
    log_every_episodes: int = 5


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    features: FeatureConfig
    environment: EnvironmentConfig
    splits: SplitConfig
    experiment: ExperimentOptions
    rl: RLConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        """Build a typed experiment config from a dictionary payload."""
        feature_payload = dict(payload.get("features", {}))
        environment_payload = dict(payload.get("environment", {}))

        sentiment_variant = feature_payload.get("sentiment_variant")
        state_mode = feature_payload.get("state_mode")
        sentiment_imputation_mode = feature_payload.get("sentiment_imputation_mode")
        feature_payload["sentiment_variant"] = normalize_sentiment_variant(
            sentiment_variant=sentiment_variant,
            state_mode=state_mode,
            sentiment_imputation_mode=sentiment_imputation_mode,
        )
        feature_payload["state_mode"] = state_mode_from_sentiment_variant(feature_payload["sentiment_variant"])
        feature_payload["sentiment_imputation_mode"] = feature_payload["sentiment_variant"]
        environment_payload["reward_mode"] = normalize_reward_mode(environment_payload.get("reward_mode", "profit"))

        return cls(
            data=DataConfig(**payload.get("data", {})),
            features=FeatureConfig(**feature_payload),
            environment=EnvironmentConfig(**environment_payload),
            splits=SplitConfig(**payload.get("splits", {})),
            experiment=ExperimentOptions(**payload.get("experiment", {})),
            rl=RLConfig(**payload.get("rl", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment settings from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def normalize_reward_mode(reward_mode: str) -> str:
    """Map user-facing and legacy reward names to the canonical internal label."""
    normalized = reward_mode.strip().lower()
    if normalized == "profit":
        return "profit"
    if normalized in {"sharpe", "differential_sharpe"}:
        return "sharpe"
    raise ValueError(f"Unsupported reward mode: {reward_mode}")


def normalize_sentiment_variant(
    sentiment_variant: str | None,
    state_mode: str | None = None,
    sentiment_imputation_mode: str | None = None,
) -> str:
    """Map new and legacy sentiment settings to the canonical variant label."""
    if sentiment_variant is not None:
        normalized = sentiment_variant.strip().lower()
        if normalized in {"none", "zero", "decay"}:
            return normalized
        raise ValueError(f"Unsupported sentiment variant: {sentiment_variant}")

    if sentiment_imputation_mode is not None:
        normalized = sentiment_imputation_mode.strip().lower()
        if normalized in {"none", "zero", "decay"}:
            return normalized
        raise ValueError(f"Unsupported sentiment mode: {sentiment_imputation_mode}")

    if state_mode == "price_only":
        return "none"
    if state_mode == "price_sentiment":
        return "zero"
    return "none"


def state_mode_from_sentiment_variant(sentiment_variant: str) -> str:
    """Derive the legacy state-mode label from the sentiment variant."""
    return "price_only" if sentiment_variant == "none" else "price_sentiment"
