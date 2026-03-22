"""Episode runners for comparing strategy performance."""

from __future__ import annotations

import pandas as pd

from ..envs import TradingEnvironment
from ..strategies import BaseStrategy


def run_strategy_episode(strategy: BaseStrategy, environment: TradingEnvironment) -> pd.DataFrame:
    """Run one full episode and return the environment history as a frame."""
    strategy.reset()
    observation = environment.reset()
    done = False

    while not done:
        action = strategy.select_action(observation)
        step = environment.step(action)
        observation = step.observation
        done = step.done

    return pd.DataFrame(environment.history)
