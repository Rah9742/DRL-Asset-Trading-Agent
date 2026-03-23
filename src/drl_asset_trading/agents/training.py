"""Training and evaluation helpers for the Double DQN agent."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from ..artifacts import experiment_manifest, write_json_artifact
from ..config import ExperimentConfig
from ..data import split_by_dates
from ..envs import TradingEnvironment
from ..evaluation import compute_performance_metrics, run_strategy_episode
from ..evaluation.benchmarks import derive_feature_columns
from ..evaluation.scaling import save_feature_scaler, scale_dataset_splits
from .double_dqn import DoubleDQNAgent


def set_global_seeds(seed: int) -> None:
    """Seed the main randomness sources used by the RL pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_agent(
    agent: DoubleDQNAgent,
    dataset: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run greedy evaluation for a trained agent on a dataset split."""
    environment = TradingEnvironment(
        market_data=dataset,
        feature_columns=derive_feature_columns(dataset),
        config=config.environment,
    )
    history = run_strategy_episode(agent, environment)
    metrics = compute_performance_metrics(
        history["portfolio_value"],
        annualization_factor=config.environment.annualization_factor,
    )
    return history, metrics


def _validation_selection_key(validation_metrics: dict[str, float], validation_metric: str) -> tuple[float, ...]:
    """Build the checkpoint-selection key from validation metrics."""
    if validation_metric == "sharpe_ratio_then_cumulative_return":
        return (
            float(validation_metrics["sharpe_ratio"]),
            float(validation_metrics["cumulative_return"]),
        )
    if validation_metric in validation_metrics:
        return (float(validation_metrics[validation_metric]),)
    raise ValueError(f"Unsupported validation metric: {validation_metric}")


def _format_validation_value(selection_key: tuple[float, ...]) -> str:
    """Format the checkpoint-selection key for logs."""
    if len(selection_key) == 1:
        return f"{selection_key[0]:.6f}"
    return ", ".join(f"{value:.6f}" for value in selection_key)


def train_double_dqn(
    dataset: pd.DataFrame,
    config: ExperimentConfig,
    dataset_name: str,
    run_name: str | None = None,
) -> dict[str, object]:
    """Train a Double DQN agent and select the best checkpoint on validation performance."""
    set_global_seeds(config.experiment.random_seed)
    splits = split_by_dates(dataset, config.splits)
    feature_columns = derive_feature_columns(dataset)
    scaled_splits, scaler = scale_dataset_splits(splits, feature_columns)

    train_env = TradingEnvironment(
        market_data=scaled_splits["train"],
        feature_columns=feature_columns,
        config=config.environment,
    )
    observation_dim = len(train_env.reset())
    action_dim = 3

    agent = DoubleDQNAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        config=config.rl,
        seed=config.experiment.random_seed,
    )

    run_name = run_name or dataset_name
    checkpoint_dir = Path(config.rl.checkpoint_dir) / run_name
    results_dir = Path(config.rl.results_dir) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}"
    scaler_path = save_feature_scaler(scaler, results_dir / f"{stem}_feature_scaler.json")
    manifest_path = write_json_artifact(
        results_dir / f"{stem}_manifest.json",
        experiment_manifest(
            config,
            dataset_name=dataset_name,
            run_name=run_name,
            scaler_path=scaler_path,
        ),
    )

    best_validation_key: tuple[float, ...] | None = None
    best_checkpoint_path = checkpoint_dir / f"{stem}_best.pt"
    training_log_rows: list[dict[str, float | int]] = []

    for episode in range(1, config.rl.training_episodes + 1):
        observation = train_env.reset()
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []

        while not done:
            action = agent.act(observation, explore=True)
            step = train_env.step(action)
            agent.store_transition(observation, action, step.reward, step.observation, step.done)
            agent.step()

            if agent.total_steps >= config.rl.warmup_steps:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)

            observation = step.observation
            episode_reward += step.reward
            done = step.done

        validation_history, validation_metrics = evaluate_agent(agent, scaled_splits["validation"], config)
        validation_key = _validation_selection_key(validation_metrics, config.rl.validation_metric)

        average_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        training_log_rows.append(
            {
                "episode": episode,
                "seed": config.experiment.random_seed,
                "run_name": run_name,
                "reward_mode": config.environment.reward_mode,
                "sentiment_variant": config.features.sentiment_variant,
                "differential_sharpe_eta": config.environment.differential_sharpe_eta,
                "differential_sharpe_warmup_steps": config.environment.differential_sharpe_warmup_steps,
                "differential_sharpe_min_variance": config.environment.differential_sharpe_min_variance,
                "episode_reward": episode_reward,
                "average_loss": average_loss,
                "epsilon": agent.epsilon,
                "validation_cumulative_return": validation_metrics["cumulative_return"],
                "validation_annualized_return": validation_metrics["annualized_return"],
                "validation_annualized_volatility": validation_metrics["annualized_volatility"],
                "validation_sharpe_ratio": validation_metrics["sharpe_ratio"],
                "validation_max_drawdown": validation_metrics["max_drawdown"],
                "validation_selector": config.rl.validation_metric,
            }
        )

        if config.rl.log_every_episodes > 0 and (
            episode == 1
            or episode == config.rl.training_episodes
            or episode % config.rl.log_every_episodes == 0
        ):
            print(
                f"[{run_name}] episode {episode}/{config.rl.training_episodes} "
                f"reward={episode_reward:.6f} loss={average_loss:.6f} "
                f"epsilon={agent.epsilon:.4f} "
                f"val_{config.rl.validation_metric}={_format_validation_value(validation_key)}",
                flush=True,
            )

        if best_validation_key is None or validation_key > best_validation_key:
            best_validation_key = validation_key
            agent.save_checkpoint(
                best_checkpoint_path,
                metadata={
                    "episode": episode,
                    "validation_metrics": validation_metrics,
                    "validation_selector": config.rl.validation_metric,
                    "validation_selection_key": list(validation_key),
                    "dataset": dataset_name,
                    "run_name": run_name,
                    "seed": config.experiment.random_seed,
                    "reward_mode": config.environment.reward_mode,
                    "sentiment_variant": config.features.sentiment_variant,
                    "differential_sharpe_eta": config.environment.differential_sharpe_eta,
                    "differential_sharpe_warmup_steps": config.environment.differential_sharpe_warmup_steps,
                    "differential_sharpe_min_variance": config.environment.differential_sharpe_min_variance,
                },
            )
            validation_history.to_csv(
                results_dir / f"{config.data.ticker}_{config.data.start_date}_{config.data.end_date}_best_validation_history.csv",
                index=False,
            )
            print(
                f"[{run_name}] new best checkpoint at episode {episode} "
                f"with val_{config.rl.validation_metric}={_format_validation_value(validation_key)}",
                flush=True,
            )

    training_log = pd.DataFrame(training_log_rows)
    training_log_path = results_dir / f"{stem}_training_log.csv"
    training_log.to_csv(training_log_path, index=False)

    best_metadata = agent.load_checkpoint(best_checkpoint_path)
    split_metrics_rows: list[dict[str, object]] = []
    split_histories: dict[str, pd.DataFrame] = {}

    for split_name, split_frame in scaled_splits.items():
        history, metrics = evaluate_agent(agent, split_frame, config)
        split_metrics_rows.append(
            {
                "seed": config.experiment.random_seed,
                "run_name": run_name,
                "reward_mode": config.environment.reward_mode,
                "sentiment_variant": config.features.sentiment_variant,
                "differential_sharpe_eta": config.environment.differential_sharpe_eta,
                "differential_sharpe_warmup_steps": config.environment.differential_sharpe_warmup_steps,
                "differential_sharpe_min_variance": config.environment.differential_sharpe_min_variance,
                "dataset": split_name,
                "strategy": "double_dqn",
                **metrics,
            }
        )
        split_histories[split_name] = history
        history.to_csv(
            results_dir / f"{stem}_{split_name}_history.csv",
            index=False,
        )

    split_metrics = pd.DataFrame(split_metrics_rows)
    split_metrics_path = results_dir / f"{stem}_metrics.csv"
    split_metrics.to_csv(split_metrics_path, index=False)

    return {
        "agent": agent,
        "best_checkpoint_path": best_checkpoint_path,
        "training_log_path": training_log_path,
        "metrics_path": split_metrics_path,
        "scaler_path": scaler_path,
        "manifest_path": manifest_path,
        "split_metrics": split_metrics,
        "split_histories": split_histories,
        "best_metadata": best_metadata,
    }
