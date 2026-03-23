# Deep Reinforcement Learning Asset Trading Agent

Coursework-oriented codebase for evaluating reward engineering and state augmentation in a single-asset deep reinforcement learning trading setting.

## Proposed Structure

```text
.
├── AGENTS.md
├── README.md
├── configs/
│   └── baseline_experiment.json
├── requirements.txt
└── src/
    └── drl_asset_trading/
        ├── __init__.py
        ├── config.py
        ├── data/
        ├── envs/
        ├── evaluation/
        ├── features/
        └── strategies/
```

## Assumptions To Fix Early

- Data source: `yfinance` by default, with Alpha Vantage reserved as an optional fallback provider.
- Action space: long/flat discrete actions (`hold`, `buy`, `sell`) for the initial baseline.
- Reward baseline: portfolio return per step; risk-aware reward to be added as a separate variant.
- Splits: explicit train/validation/test date windows to avoid leakage from random sampling.

## Initial Phases

1. Scaffold configuration, data loading, feature engineering, environment, strategy interfaces, and evaluation utilities.
2. Add deterministic experiment runners and baseline heuristic strategies.
3. Implement DQN/Double DQN agent training and validation.
4. Add reward-engineering and state-augmentation variants, then compare out-of-sample performance.

## Experiment Naming

The experiment interface uses two explicit axes:

- `reward_mode`: `profit` or `sharpe`
- `sentiment_variant`: `none`, `zero`, or `decay`

Run names are built directly from those choices, for example `profit_none` or `sharpe_decay`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

## Docs

- [`docs/ablation_variants.md`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/docs/ablation_variants.md)
- [`docs/reward_engineering.md`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/docs/reward_engineering.md)
