# Running Experiments

This guide covers:

- how to run each sub-experiment entry point
- how the config file controls data, features, rewards, splits, and training
- when you need to rebuild cached datasets

## Config Structure

The main config file is a JSON document with six top-level sections:

- `data`
- `features`
- `environment`
- `splits`
- `experiment`
- `rl`

Example:

```json
{
  "data": {
    "provider": "yfinance",
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "interval": "1d"
  },
  "features": {
    "sentiment_variant": "none",
    "include_returns": true,
    "include_log_returns": true,
    "include_momentum": true,
    "include_volatility": true,
    "include_rsi": true,
    "include_cyclical_time_features": true,
    "lookback_window": 14,
    "sentiment_lag_days": 1,
    "sentiment_short_window": 3,
    "sentiment_long_window": 7,
    "sentiment_fill_value": 0.0,
    "sentiment_decay_rate": 0.25
  },
  "environment": {
    "initial_cash": 10000.0,
    "transaction_cost": 0.001,
    "reward_mode": "profit",
    "annualization_factor": 252,
    "differential_sharpe_eta": 0.005
  },
  "splits": {
    "train_end": "2020-12-31",
    "validation_end": "2022-12-31",
    "test_end": "2024-12-31"
  },
  "experiment": {
    "random_seed": 42,
    "seed_values": [42, 43, 44, 45]
  },
  "rl": {
    "algorithm": "double_dqn",
    "hidden_dim": 128,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "training_episodes": 50,
    "validation_metric": "sharpe_ratio",
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0
  }
}
```

## What Each Config Section Controls

### `data`

Use this section to control the raw market and sentiment date range:

- `ticker`: target asset symbol
- `start_date`, `end_date`: raw download and processed dataset range
- `provider`: price provider, usually `yfinance`
- `interval`: currently daily only in the main setup

If you change `ticker`, `start_date`, or `end_date`, rerun data loading and feature building.

### `features`

Use this section to control engineered state inputs:

- `sentiment_variant`: `none`, `sparse`, `decay`
- `lookback_window`: used by momentum, volatility, and RSI
- `include_cyclical_time_features`: toggles cyclical time inputs
- `sentiment_lag_days`: shifts sentiment forward before joining to price
- `sentiment_short_window`, `sentiment_long_window`: control rolling sentiment features
- `sentiment_fill_value`: sparse-mode fill value
- `sentiment_decay_rate`: decay strength for `decay` mode

Changing feature settings means you should rebuild processed datasets.

### `environment`

Use this section for trading and reward assumptions:

- `initial_cash`
- `transaction_cost`
- `reward_mode`: `profit` or `sharpe`
- `annualization_factor`
- `differential_sharpe_eta`
- `differential_sharpe_warmup_steps`
- `differential_sharpe_min_variance`

### `splits`

This controls the date-based experiment windows:

- `train_end`
- `validation_end`
- `test_end`

The code expects all three splits to be non-empty. Keep `train_end < validation_end < test_end`.

### `experiment`

Use this section for reproducibility:

- `random_seed`: single-run seed
- `seed_values`: list used by multi-seed comparison runners

### `rl`

Use this section for optimizer and model-selection settings:

- `hidden_dim`
- `learning_rate`
- `weight_decay`
- `batch_size`
- `buffer_capacity`
- `training_episodes`
- `warmup_steps`
- `target_update_frequency`
- `validation_metric`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `checkpoint_dir`
- `results_dir`

## Main Sub-Experiment Commands

### 1. Full Pipeline

Runs price load, sentiment load, feature building, then a comparison runner.

```bash
python3 -m drl_asset_trading.main \
  --config configs/baseline_experiment.json \
  --comparison-mode full \
  --seeds 42,43,44,45
```

Useful overrides:

- `--ticker AAPL`
- `--comparison-mode profit_sentiment`
- `--force-price-download`
- `--force-sentiment-download`

### 2. Price Loading Only

```bash
python3 -m drl_asset_trading.data.run_price_loader \
  --config configs/baseline_experiment.json
```

Use this when:

- you changed `data.ticker`
- you changed `data.start_date` or `data.end_date`
- you want to refresh the cached raw price CSV

### 3. Sentiment Loading Only

```bash
python3 -m drl_asset_trading.data.run_sentiment_loader \
  --config configs/baseline_experiment.json
```

Use this when:

- you changed ticker or date range
- you changed sentiment query settings
- you want fresh raw or interim sentiment artifacts

### 4. Feature Building Only

```bash
python3 -m drl_asset_trading.features.run_feature_builder \
  --config configs/baseline_experiment.json
```

This rebuilds:

- `price`
- `price_sentiment_sparse`
- `price_sentiment_decay`

It also refreshes the feature diagnostics report in `reports/<TICKER>/..._feature_diagnostics.json`.

### 5. Benchmarks Only

Heuristic baselines for a chosen processed dataset:

```bash
python3 -m drl_asset_trading.evaluation.run_benchmarks \
  --config configs/baseline_experiment.json \
  --dataset price
```

Dataset choices:

- `price`
- `price_sentiment_sparse`
- `price_sentiment_decay`

### 6. Single DDQN Experiment

Run one reward/sentiment combination:

```bash
python3 -m drl_asset_trading.experiments.run_ablation \
  --config configs/baseline_experiment.json \
  --reward-mode profit \
  --sentiment-variant decay
```

Examples:

- profit reward, no sentiment:
  - `--reward-mode profit --sentiment-variant none`
- profit reward, sparse sentiment:
  - `--reward-mode profit --sentiment-variant sparse`
- differential Sharpe reward, decay sentiment:
  - `--reward-mode sharpe --sentiment-variant decay`

### 7. Full Comparison Suite

Runs:

- buy-and-hold benchmark
- random benchmark
- `profit_none`
- `profit_sparse`
- `profit_decay`
- `sharpe_none`
- `sharpe_sparse`
- `sharpe_decay`

```bash
python3 -m drl_asset_trading.experiments.run_full_comparison \
  --config configs/baseline_experiment.json \
  --seeds 42,43,44,45
```

### 8. Profit-Only Sentiment Comparison

Useful when you want to isolate the effect of state augmentation under a fixed reward:

```bash
python3 -m drl_asset_trading.experiments.run_profit_sentiment_comparison \
  --config configs/baseline_experiment.json \
  --seeds 42,43,44,45
```

### 9. Differential Sharpe Eta Sweep

```bash
python3 -m drl_asset_trading.experiments.run_differential_sharpe_eta_sweep \
  --config configs/baseline_experiment.json \
  --seeds 42,43,44,45 \
  --sentiment-variants none decay
```

Use this when tuning `environment.differential_sharpe_eta`.

## When To Rebuild What

Rerun price loading if you changed:

- `data.ticker`
- `data.start_date`
- `data.end_date`
- price provider settings

Rerun sentiment loading if you changed:

- ticker or date range
- sentiment query options

Rerun feature building if you changed:

- anything in `features`
- ticker or date range
- sentiment source artifacts

You do not need to rebuild raw data just because you changed:

- `environment`
- `splits`
- `experiment`
- `rl`

Those affect training/evaluation, not raw downloads.

## Recommended Workflow

For a clean new experiment:

1. edit a config JSON in `configs/`
2. run the feature-building stage if data or features changed
3. run a single DDQN combination first to sanity-check
4. run the multi-seed comparison you actually want to compare
5. inspect `reports/`, `results/`, and the feature diagnostics artifact

## Notes

- Feature scaling is fit on the train split only and persisted per run.
- The best checkpoint is selected from validation metrics only.
- Test metrics are reported after loading the best validation checkpoint.
- If you change the feature set but reuse old processed CSVs, your experiment will not reflect the new config.
