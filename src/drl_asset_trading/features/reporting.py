"""Feature diagnostics for state modes and leakage-sensitive preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..artifacts import write_json_artifact
from ..config import ExperimentConfig
from ..data import split_by_dates
from .engineering import BASE_SENTIMENT_COLUMNS

NON_FEATURE_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"}
STATE_MODE_LABELS = {
    "price": "price_only",
    "price_sentiment_sparse": "price_sentiment_sparse",
    "price_sentiment_decay": "price_sentiment_decay",
}
STATIC_SENTIMENT_FEATURE_COLUMNS = set(BASE_SENTIMENT_COLUMNS) | {
    "mean_overall_sentiment",
    "weighted_ticker_sentiment",
    "days_since_last_news",
}


def default_feature_diagnostics_path(config: ExperimentConfig) -> Path:
    """Return the default path for the feature diagnostics artifact."""
    stem = f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}"
    return Path("reports") / config.data.ticker / f"{stem}_feature_diagnostics.json"


def save_feature_diagnostics(
    datasets: dict[str, pd.DataFrame],
    config: ExperimentConfig,
    path: str | Path | None = None,
) -> Path:
    """Build and persist a feature diagnostics report for all available state modes."""
    output_path = Path(path) if path is not None else default_feature_diagnostics_path(config)
    return write_json_artifact(output_path, build_feature_diagnostics(datasets=datasets, config=config))


def build_feature_diagnostics(
    datasets: dict[str, pd.DataFrame],
    config: ExperimentConfig,
) -> dict[str, object]:
    """Summarize feature usage and train-only redundancy checks for each state mode."""
    report: dict[str, object] = {
        "ticker": config.data.ticker,
        "start_date": config.data.start_date,
        "end_date": config.data.end_date,
        "split_boundaries": {
            "train_end": config.splits.train_end,
            "validation_end": config.splits.validation_end,
            "test_end": config.splits.test_end,
        },
        "leakage_controls": {
            "feature_scaling_fit_split": "train",
            "feature_scaler_persisted_per_run": True,
            "redundant_sentiment_analysis_fit_split": "train",
            "price_feature_engineering": "causal rolling, return, and cyclical time features only",
            "sentiment_lag_days": config.features.sentiment_lag_days,
            "sentiment_decay_uses_only_past_observations": True,
            "validation_or_test_used_for_scaling": False,
            "validation_or_test_used_for_redundancy_detection": False,
        },
        "state_modes": {},
    }

    state_modes: dict[str, object] = {}
    for dataset_name, dataset in datasets.items():
        feature_columns = [column for column in dataset.columns if column not in NON_FEATURE_COLUMNS]
        sentiment_feature_columns = [column for column in feature_columns if _is_sentiment_feature_name(column)]
        price_feature_columns = [column for column in feature_columns if not _is_sentiment_feature_name(column)]
        splits = split_by_dates(dataset, config.splits)
        redundant_pairs = _identify_redundant_sentiment_features(
            train_split=splits["train"],
            sentiment_feature_columns=sentiment_feature_columns,
        )

        state_modes[STATE_MODE_LABELS.get(dataset_name, dataset_name)] = {
            "dataset_name": dataset_name,
            "feature_columns": feature_columns,
            "price_feature_columns": price_feature_columns,
            "sentiment_feature_columns": sentiment_feature_columns,
            "split_row_counts": {name: int(len(frame)) for name, frame in splits.items()},
            "redundant_sentiment_feature_pairs": redundant_pairs,
        }

    report["state_modes"] = state_modes
    return report


def _is_sentiment_feature_name(feature_name: str) -> bool:
    """Return True when a feature belongs to the sentiment modality."""
    return feature_name in STATIC_SENTIMENT_FEATURE_COLUMNS or feature_name.startswith(
        ("sentiment_mean_", "sentiment_diff_", "sentiment_window_spread_")
    )


def _identify_redundant_sentiment_features(
    train_split: pd.DataFrame,
    sentiment_feature_columns: list[str],
    threshold: float = 0.95,
) -> list[dict[str, float | str]]:
    """Flag highly correlated sentiment feature pairs using the train split only."""
    if len(sentiment_feature_columns) < 2:
        return []

    sentiment_frame = train_split.loc[:, sentiment_feature_columns].copy()
    constant_columns = [
        column
        for column in sentiment_feature_columns
        if float(sentiment_frame[column].std(ddof=0)) == 0.0
    ]
    non_constant_columns = [column for column in sentiment_feature_columns if column not in constant_columns]

    redundant_pairs: list[dict[str, float | str]] = [
        {
            "feature_a": column,
            "feature_b": "",
            "reason": "constant_on_train_split",
            "absolute_correlation": 1.0,
        }
        for column in constant_columns
    ]

    if len(non_constant_columns) < 2:
        return redundant_pairs

    correlation = sentiment_frame.loc[:, non_constant_columns].corr().abs()
    for index, feature_a in enumerate(non_constant_columns):
        for feature_b in non_constant_columns[index + 1 :]:
            absolute_correlation = float(correlation.loc[feature_a, feature_b])
            if absolute_correlation >= threshold:
                redundant_pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "reason": "high_train_correlation",
                        "absolute_correlation": absolute_correlation,
                    }
                )
    return redundant_pairs
