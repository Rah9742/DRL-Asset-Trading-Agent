# Experiment Axes

This project names DDQN experiments using two explicit axes:

- `reward_mode`
- `sentiment_variant`

That is the public experiment interface. The older `state_only`, `both`, and `sentiment_imputation_mode` labels are treated as legacy aliases.

## Reward Mode

- `profit`
  - one-step portfolio return
  - standard return-maximization objective

- `sharpe`
  - online differential Sharpe reward
  - risk-aware objective based on exponentially weighted return moments
  - see [`reward_engineering.md`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/docs/reward_engineering.md)

## Sentiment Variant

- `none`
  - no sentiment columns in the state
  - loads the processed dataset `baseline`

- `zero`
  - sentiment columns included
  - no-news days use zero-filled sentiment values
  - loads the processed dataset `sentiment_zero`

- `decay`
  - sentiment columns included
  - no-news days carry the latest sentiment forward with exponential decay
  - includes `days_since_last_news`
  - loads the processed dataset `augmented`

## Canonical Run Names

Run names are built directly from the two axes:

- `profit_none`
- `sharpe_none`
- `profit_zero`
- `profit_decay`
- `sharpe_zero`
- `sharpe_decay`

These names are intended to be self-explanatory:

- the first part tells you the training objective
- the second part tells you which sentiment feature set was used

## Legacy Mapping

Older coursework labels still map onto the new names:

- `baseline` -> `profit_none`
- `reward_only` -> `sharpe_none`
- `state_only` -> `profit_decay`
- `both` -> `sharpe_decay`

Those aliases are only kept for backward compatibility. New runs should use `reward_mode` and `sentiment_variant` directly.

## Recommended Comparisons

Clean comparisons now read naturally:

- `profit_none` vs `profit_zero`
- `profit_none` vs `profit_decay`
- `profit_zero` vs `profit_decay`
- `sharpe_none` vs `sharpe_decay`

This keeps the experimental story tied to the actual knobs the code is using.

## Single-Call Profit Comparison

To compare the fixed-`profit` runs across all three sentiment settings in one call:

```bash
python -m drl_asset_trading.experiments.run_profit_sentiment_comparison \
  --config configs/baseline_experiment.json
```

That runs:

- `profit_none`
- `profit_zero`
- `profit_decay`
