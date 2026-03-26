# Reward Engineering Notes

## Baseline Reward

The baseline reward is the one-step portfolio return:

```text
r_t = V_t / V_{t-1} - 1
```

where:

- `V_t` is the portfolio value after the action and transaction costs at time `t`
- `V_{t-1}` is the previous portfolio value

## Sharpe Reward

The `sharpe` reward mode uses an online differential Sharpe approximation based on exponentially weighted return moments.

Let:

```text
A_t = A_{t-1} + eta * (r_t - A_{t-1})
B_t = B_{t-1} + eta * (r_t^2 - B_{t-1})
```

where:

- `r_t` is the one-step portfolio return
- `A_t` is the exponentially weighted first moment of returns
- `B_t` is the exponentially weighted second moment of returns
- `eta` is `environment.differential_sharpe_eta`

The implemented reward uses the previous-step moments:

```text
D_t =
    [B_{t-1}(r_t - A_{t-1}) - 0.5 A_{t-1}(r_t^2 - B_{t-1})]
    / max(B_{t-1} - A_{t-1}^2, epsilon)^(3/2)
```

where `epsilon` is `environment.differential_sharpe_epsilon`.

## Warm-Up And Stability Rule

The environment emits plain portfolio return instead of differential Sharpe while either condition holds:

- fewer than `differential_sharpe_warmup_steps` returns have been absorbed
- the estimated return variance is below `differential_sharpe_min_variance`

This is separate from `rl.warmup_steps`.

- `differential_sharpe_warmup_steps` controls when the environment may emit Sharpe-style rewards
- `rl.warmup_steps` controls when the agent may start replay-buffer gradient updates

## Eta Tuning

For daily data, the checked-in working default is `0.005`, with the recommended comparison grid stored in config:

- `0.001`
- `0.005`
- `0.01`

To run the eta sweep:

```bash
python -m drl_asset_trading.experiments.run_differential_sharpe_eta_sweep \
  --config configs/baseline_experiment.json
```

Optional overrides:

- `--etas 0.001 0.005 0.01`
- `--seeds 42,43,44,45,46`
- `--sentiment-variants none decay`

## Checkpoint Selection And Early Stopping

The training loop selects checkpoints on the validation split using `rl.validation_metric`.

Supported options are:

- `sharpe_ratio`
- `cumulative_return`
- `sharpe_ratio_then_cumulative_return`

The same metric is also used for early stopping.

Relevant controls:

- `rl.validation_metric`
- `rl.early_stopping_patience`
- `rl.early_stopping_min_delta`

`sharpe_ratio_then_cumulative_return` ranks checkpoints by validation Sharpe ratio first and uses validation cumulative return as the tiebreak.

## Regularisation And Overfitting Controls

The DDQN training pipeline now includes several controls beyond reward design:

- feature scaling fit on the train split only
- persisted per-run scaler artifacts
- validation-based checkpointing
- early stopping on validation performance
- optional Adam `weight_decay`
- explicit train/validation/test metric logging

Important leakage rules:

- validation and test data are not used to fit the scaler
- test data are not used for checkpoint selection or early stopping
- redundant-feature diagnostics are computed from the train split only

## Interpretation

- `profit` reward encourages pure return maximisation
- `sharpe` encourages return quality by penalising unstable return paths through the Sharpe-style denominator

Reward changes should be interpreted jointly with the overfitting controls above. A stronger reward function is not useful if model selection still favors unstable checkpoints.

## Location In Code

Main implementation points:

- [`trading_env.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/envs/trading_env.py)
- [`training.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/agents/training.py)
- [`scaling.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/evaluation/scaling.py)
- [`config.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/config.py)
- [`run_differential_sharpe_eta_sweep.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/experiments/run_differential_sharpe_eta_sweep.py)
