# Ablation Variants

This project uses a 2x2 ablation design with two experiment axes:

- `state_mode`: what information is given to the agent
- `reward_mode`: what objective the agent is trained to optimise

## Experiment Axes

### State Mode

- `price_only`
  - the agent sees only price-derived and technical-indicator features
  - this is the plain trading-state baseline

- `price_sentiment`
  - the agent sees the same price/technical features plus sentiment features
  - sentiment is merged into the same daily dataframe as the market features
  - sentiment can use:
    - `sentiment_imputation_mode = "zero"`
    - `sentiment_imputation_mode = "decay"`

### Reward Mode

- `profit`
  - the reward is one-step portfolio return
  - this is the standard return-maximisation baseline

- `differential_sharpe`
  - the reward is an online differential Sharpe approximation
  - this is the reward-engineering variant
  - see [`reward_engineering.md`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/docs/reward_engineering.md)

## Four Variants

### 1. Baseline

```text
state_mode = price_only
reward_mode = profit
```

Meaning:

- no sentiment in the state
- no risk-aware reward shaping
- this is the plain Double DQN trading baseline

### 2. Reward Only

```text
state_mode = price_only
reward_mode = differential_sharpe
```

Meaning:

- same state as the baseline
- only the reward changes
- isolates the effect of reward engineering

### 3. State Only

```text
state_mode = price_sentiment
reward_mode = profit
```

Meaning:

- reward stays the same as the baseline
- only the state is enriched with sentiment features
- isolates the effect of state augmentation

### 4. Both

```text
state_mode = price_sentiment
reward_mode = differential_sharpe
```

Meaning:

- sentiment is included in the state
- the reward is also risk-aware
- this is the combined extension

## Baseline vs Both

This is the main difference:

- `baseline`
  - learns from price/technical signals only
  - is trained to maximise raw step-by-step trading return

- `both`
  - learns from price/technical signals plus sentiment features
  - is trained with a differential Sharpe reward that prefers better risk-adjusted return dynamics

So `both` changes two things at once:

1. the information available to the agent
2. the optimisation target used during training

That is why the intermediate variants are important:

- `reward_only` tells you whether reward engineering helps on its own
- `state_only` tells you whether sentiment-state augmentation helps on its own
- `both` tells you whether combining both changes helps more than either alone

## Sentiment State Variants

When `state_mode = price_sentiment`, the sentiment features can be constructed in two ways:

- `zero`
  - no-news days set sentiment features to zero
  - `news_count` remains available so the model can distinguish no-news from neutral sentiment

- `decay`
  - the latest available sentiment is carried forward with exponential decay
  - `days_since_last_news` is included as an explicit feature

In this project:

- `price_only` maps to the processed dataset `baseline`
- `price_sentiment + zero` maps to `sentiment_zero`
- `price_sentiment + decay` maps to `augmented`

## Interpretation

This setup gives a clean interpretation:

- baseline vs reward_only: effect of reward engineering
- baseline vs state_only: effect of state augmentation
- baseline vs both: effect of combining both changes

That makes the final experimental story easy to explain in a coursework report.
