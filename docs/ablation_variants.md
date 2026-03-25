# Experiment Axes

This project names DDQN experiments with two explicit axes:

- `reward_mode`
- `sentiment_variant`

That is the public experiment interface.

## Reward Mode

- `profit`
  - one-step portfolio return
  - pure return objective

- `sharpe`
  - online differential Sharpe reward
  - risk-aware objective based on exponentially weighted return moments
  - see [`reward_engineering.md`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/docs/reward_engineering.md)

## Sentiment Variant

- `none`
  - loads dataset `price`
  - uses the engineered daily price/time state only
  - DDQN falls back to the price-only MLP path

- `sparse`
  - loads dataset `price_sentiment_sparse`
  - uses the same daily price/time state plus lagged daily sentiment
  - no-news days are zero-filled

- `decay`
  - loads dataset `price_sentiment_decay`
  - uses the same daily price/time state plus lagged daily sentiment
  - no-news days carry the latest sentiment forward with exponential decay
  - includes `days_since_last_news`

## Shared Price Feature Set

All three dataset variants share the same engineered daily price/time features:

- `return_1`
- `log_return_1`
- `momentum_{lookback}`
- `volatility_{lookback}`
- `rsi_{lookback}`
- `volume_change_1`
- cyclical `day_of_week_sin/cos`
- cyclical `day_of_year_sin/cos`

## Sentiment Feature Set

When sentiment is enabled, the model-facing sentiment branch uses:

- `news_count`
- `mean_ticker_sentiment`
- `mean_ticker_relevance`
- `sentiment_std`
- `sentiment_mean_3`
- `sentiment_mean_7`
- `sentiment_diff_1`
- `sentiment_window_spread_3_7`

Notes:

- sentiment is lagged by `features.sentiment_lag_days`
- `mean_overall_sentiment` and `weighted_ticker_sentiment` are still available in the interim daily sentiment CSV but are no longer included in the processed state
- feature diagnostics flag highly correlated sentiment features on the train split only

## Model Interpretation

The DDQN now treats price and sentiment as separate modalities when sentiment columns are present:

- price features plus position -> price embedding
- sentiment features -> sentiment embedding
- each embedding uses `Linear -> ELU`
- the fused representation feeds the Q-value head

That means the ablation axis now changes both the feature set and, when sentiment is present, the multimodal routing inside the network.

## Canonical Run Names

Run names are built directly from the two axes:

- `profit_none`
- `profit_sparse`
- `profit_decay`
- `sharpe_none`
- `sharpe_sparse`
- `sharpe_decay`

## Recommended Comparisons

The cleanest comparisons are:

- `profit_none` vs `profit_sparse`
- `profit_none` vs `profit_decay`
- `profit_sparse` vs `profit_decay`
- `sharpe_none` vs `sharpe_decay`

For stability, prefer multiple seeds rather than a single-seed headline result.

## Commands

Full comparison:

```bash
python -m drl_asset_trading.experiments.run_full_comparison \
  --config configs/baseline_experiment.json \
  --seeds 42,43,44,45
```

Profit-only sentiment comparison:

```bash
python -m drl_asset_trading.experiments.run_profit_sentiment_comparison \
  --config configs/baseline_experiment.json \
  --seeds 42,43,44,45
```
