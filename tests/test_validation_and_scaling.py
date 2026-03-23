from __future__ import annotations

import unittest

import pandas as pd

from drl_asset_trading.agents.training import _validation_selection_key
from drl_asset_trading.evaluation.scaling import apply_feature_scaler, fit_feature_scaler, scale_dataset_splits


class ValidationSelectionTests(unittest.TestCase):
    def test_composite_validation_key_orders_by_sharpe_then_return(self) -> None:
        metrics = {"sharpe_ratio": 1.25, "cumulative_return": 0.40}
        self.assertEqual(
            _validation_selection_key(metrics, "sharpe_ratio_then_cumulative_return"),
            (1.25, 0.40),
        )


class FeatureScalingTests(unittest.TestCase):
    def test_scaler_is_fit_on_train_split_only(self) -> None:
        train = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0], "feature_b": [10.0, 10.0, 10.0]})
        validation = pd.DataFrame({"feature_a": [4.0], "feature_b": [12.0]})
        test = pd.DataFrame({"feature_a": [5.0], "feature_b": [8.0]})

        scaled_splits, scaler = scale_dataset_splits(
            {"train": train, "validation": validation, "test": test},
            ["feature_a", "feature_b"],
        )

        self.assertAlmostEqual(scaler.means["feature_a"], 2.0)
        self.assertAlmostEqual(scaler.stds["feature_a"], (2.0 / 3.0) ** 0.5)
        self.assertEqual(scaler.stds["feature_b"], 1.0)
        self.assertAlmostEqual(float(scaled_splits["train"]["feature_a"].mean()), 0.0, places=7)
        self.assertAlmostEqual(float(scaled_splits["validation"]["feature_a"].iloc[0]), 2.449489742783178, places=7)

    def test_apply_feature_scaler_preserves_non_feature_columns(self) -> None:
        dataset = pd.DataFrame({"Date": ["2024-01-01"], "feature_a": [4.0]})
        scaler = fit_feature_scaler(pd.DataFrame({"feature_a": [2.0, 4.0, 6.0]}), ["feature_a"])
        scaled = apply_feature_scaler(dataset, scaler)
        self.assertEqual(scaled["Date"].iloc[0], "2024-01-01")
        self.assertAlmostEqual(float(scaled["feature_a"].iloc[0]), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
