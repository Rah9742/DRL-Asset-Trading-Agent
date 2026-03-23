"""Helpers for persisting experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return output_path


def experiment_manifest(
    config: ExperimentConfig,
    *,
    dataset_name: str,
    run_name: str,
    scaler_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable experiment manifest."""
    manifest = {
        "config": asdict(config),
        "dataset_name": dataset_name,
        "run_name": run_name,
        "scaler_path": str(scaler_path) if scaler_path is not None else None,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _json_default(value: Any) -> Any:
    """Convert non-JSON-native values used in artifacts."""
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
