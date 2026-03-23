"""Feature scaling helpers for train/validation/test experiment splits."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class FeatureScaler:
    """Column-wise standardization statistics fit on the training split only."""

    feature_columns: list[str]
    means: dict[str, float]
    stds: dict[str, float]


def fit_feature_scaler(dataset: pd.DataFrame, feature_columns: list[str]) -> FeatureScaler:
    """Fit a simple standardization transform on the provided feature columns."""
    if not feature_columns:
        return FeatureScaler(feature_columns=[], means={}, stds={})

    means = dataset.loc[:, feature_columns].mean().to_dict()
    stds = dataset.loc[:, feature_columns].std(ddof=0).replace(0.0, 1.0).fillna(1.0).to_dict()
    return FeatureScaler(
        feature_columns=list(feature_columns),
        means={column: float(value) for column, value in means.items()},
        stds={column: float(value) for column, value in stds.items()},
    )


def apply_feature_scaler(dataset: pd.DataFrame, scaler: FeatureScaler) -> pd.DataFrame:
    """Apply a pre-fit scaler to a dataset split."""
    if not scaler.feature_columns:
        return dataset.copy()

    scaled = dataset.copy()
    for column in scaler.feature_columns:
        scaled[column] = (scaled[column] - scaler.means[column]) / scaler.stds[column]
    return scaled


def scale_dataset_splits(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[dict[str, pd.DataFrame], FeatureScaler]:
    """Fit on the train split and apply the same scaler to all splits."""
    scaler = fit_feature_scaler(splits["train"], feature_columns)
    scaled_splits = {name: apply_feature_scaler(frame, scaler) for name, frame in splits.items()}
    return scaled_splits, scaler


def save_feature_scaler(scaler: FeatureScaler, path: str | Path) -> Path:
    """Persist a fitted feature scaler for reproducible reruns."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "feature_columns": scaler.feature_columns,
                "means": scaler.means,
                "stds": scaler.stds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
