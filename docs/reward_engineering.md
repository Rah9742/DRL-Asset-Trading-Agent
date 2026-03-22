# Reward Engineering Notes

## Baseline Reward

The baseline reward is the one-step portfolio return:

```text
r_t = V_t / V_{t-1} - 1
```

where:

- `V_t` is the portfolio value at time `t`
- `V_{t-1}` is the portfolio value at the previous step

## Differential Sharpe Reward

The reward-engineering variant uses an online differential Sharpe approximation based on exponentially weighted return moments.

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

The implemented differential Sharpe reward uses the previous-step moments:

```text
D_t =
    [B_{t-1}(r_t - A_{t-1}) - 0.5 A_{t-1}(r_t^2 - B_{t-1})]
    / max(B_{t-1} - A_{t-1}^2, epsilon)^(3/2)
```

where:

- `epsilon` is a small numerical stabilizer (`differential_sharpe_epsilon`)

### Practical implementation detail

For the first few steps, the environment returns the plain portfolio return `r_t` instead of the differential Sharpe expression. This avoids unstable rewards before the running moments have enough history.

## Interpretation

- `profit` reward encourages pure return maximisation
- `differential_sharpe` encourages return quality by penalising unstable return paths through the Sharpe-style denominator

## Location in Code

The exact implementation is in:

- [`trading_env.py`](/Users/rahilshah/Development/DRL-Asset-Trading-Agent/src/drl_asset_trading/envs/trading_env.py)
