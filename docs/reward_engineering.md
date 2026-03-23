# Reward Engineering Notes

## Baseline Reward

The baseline reward is the one-step portfolio return:

```text
r_t = V_t / V_{t-1} - 1
```

where:

- `V_t` is the portfolio value at time `t`
- `V_{t-1}` is the portfolio value at the previous step
- `V_t` is computed after the environment applies the action and any transaction costs for that step

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
- `eta` is the smoothing parameter (`differential_sharpe_eta`)

The implemented reward uses the previous-step moments:

```text
D_t =
    [B_{t-1}(r_t - A_{t-1}) - 0.5 A_{t-1}(r_t^2 - B_{t-1})]
    / max(B_{t-1} - A_{t-1}^2, epsilon)^(3/2)
```

where:

- `epsilon` is a small numerical stabilizer (`differential_sharpe_epsilon`)

## Warm-Up And Stability Rule

The environment does not use the differential Sharpe expression immediately.

It returns the plain portfolio return `r_t` while either of these conditions holds:

- fewer than `differential_sharpe_warmup_steps` returns have been absorbed into the running moments
- the estimated return variance `B_{t-1} - A_{t-1}^2` is below `differential_sharpe_min_variance`

With the current checked-in config, that means:

- `differential_sharpe_warmup_steps = 20`
- `differential_sharpe_min_variance = 1e-6`

This is separate from `rl.warmup_steps`.

- `differential_sharpe_warmup_steps` controls when the environment is allowed to emit differential Sharpe rewards
- `rl.warmup_steps` controls when the agent starts gradient updates from the replay buffer

The practical effect is:

- early DSR steps use plain net portfolio return
- once the running moments have at least 20 prior observations and the variance estimate is large enough, the reward switches to the differential Sharpe expression

## Eta Tuning

`eta` controls how quickly the running return moments react to new data.

For daily data, the codebase now treats `0.005` as the default working value and keeps the recommended comparison grid in config:

- `0.001`
- `0.005`
- `0.01`

The candidate grid is stored in `environment.differential_sharpe_eta_candidates` so the sweep is reproducible from config.

To run the dedicated eta sweep for the `sharpe` reward mode across multiple sentiment variants:

```bash
python -m drl_asset_trading.experiments.run_differential_sharpe_eta_sweep \
  --config configs/baseline_experiment.json
```

Optional overrides:

- `--etas 0.001 0.005 0.01`
- `--seeds 42 43 44 45 46`
- `--sentiment-variants none decay`

## Checkpoint Selection

The training loop selects the best checkpoint on the validation split using `rl.validation_metric`.

Supported options are:

- `sharpe_ratio`
- `cumulative_return`
- `sharpe_ratio_then_cumulative_return`

The checked-in default is `sharpe_ratio`.

`sharpe_ratio_then_cumulative_return` ranks checkpoints by validation Sharpe ratio first and uses validation cumulative return only as the tiebreak.

## Interpretation

- `profit` reward encourages pure return maximisation
- `sharpe` encourages return quality by penalising unstable return paths through the Sharpe-style denominator

## Location in Code

The exact implementation is in:

- [`trading_env.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/envs/trading_env.py)
- [`config.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/config.py)
- [`run_differential_sharpe_eta_sweep.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/experiments/run_differential_sharpe_eta_sweep.py)
