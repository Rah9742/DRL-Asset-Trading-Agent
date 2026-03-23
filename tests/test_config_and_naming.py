from __future__ import annotations

import unittest

from drl_asset_trading.config import ExperimentConfig
from drl_asset_trading.experiments.run_ablation import configure_experiment
from drl_asset_trading.features.engineering import resolve_processed_dataset_name


class ConfigNormalizationTests(unittest.TestCase):
    def test_new_config_fields_normalize_to_canonical_values(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "data": {},
                "features": {"sentiment_variant": "decay"},
                "environment": {"reward_mode": "sharpe"},
                "splits": {},
                "experiment": {},
                "rl": {},
            }
        )
        self.assertEqual(config.environment.reward_mode, "sharpe")
        self.assertEqual(config.features.sentiment_variant, "decay")
        self.assertEqual(config.features.state_mode, "price_sentiment")

    def test_legacy_config_fields_still_normalize(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "data": {},
                "features": {"state_mode": "price_sentiment", "sentiment_imputation_mode": "zero"},
                "environment": {"reward_mode": "differential_sharpe"},
                "splits": {},
                "experiment": {},
                "rl": {},
            }
        )
        self.assertEqual(config.environment.reward_mode, "sharpe")
        self.assertEqual(config.features.sentiment_variant, "zero")
        self.assertEqual(config.features.state_mode, "price_sentiment")


class ExperimentNamingTests(unittest.TestCase):
    def test_configure_experiment_uses_canonical_run_name(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "data": {},
                "features": {},
                "environment": {},
                "splits": {},
                "experiment": {},
                "rl": {},
            }
        )
        configured, dataset_name, run_name = configure_experiment(config, reward_mode="profit", sentiment_variant="none")
        self.assertEqual(configured.environment.reward_mode, "profit")
        self.assertEqual(configured.features.sentiment_variant, "none")
        self.assertEqual(dataset_name, "baseline")
        self.assertEqual(run_name, "profit_none")

    def test_processed_dataset_resolution_matches_sentiment_variant(self) -> None:
        self.assertEqual(resolve_processed_dataset_name("none"), "baseline")
        self.assertEqual(resolve_processed_dataset_name("zero"), "sentiment_zero")
        self.assertEqual(resolve_processed_dataset_name("decay"), "augmented")


if __name__ == "__main__":
    unittest.main()
