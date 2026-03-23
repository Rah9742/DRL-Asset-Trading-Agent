from __future__ import annotations

import unittest

from drl_asset_trading.experiments.run_ablation import LEGACY_VARIANTS
from drl_asset_trading.experiments.run_profit_sentiment_comparison import DEFAULT_SENTIMENT_VARIANTS
from drl_asset_trading.experiments.run_full_comparison import run_full_comparison


class ComparisonInterfaceTests(unittest.TestCase):
    def test_profit_sentiment_runner_defaults_to_three_variants(self) -> None:
        self.assertEqual(DEFAULT_SENTIMENT_VARIANTS, ("none", "zero", "decay"))

    def test_legacy_variant_aliases_remain_defined(self) -> None:
        self.assertEqual(LEGACY_VARIANTS["baseline"]["reward_mode"], "profit")
        self.assertEqual(LEGACY_VARIANTS["reward_only"]["reward_mode"], "sharpe")

    def test_full_runner_still_exports_callable(self) -> None:
        self.assertTrue(callable(run_full_comparison))


if __name__ == "__main__":
    unittest.main()
