"""Double DQN agent implementation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from ..config import RLConfig
from ..strategies.base import BaseStrategy
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


class DoubleDQNAgent(BaseStrategy):
    """A small Double DQN agent for the single-asset trading environment."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: RLConfig,
        feature_columns: list[str],
        seed: int = 42,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.policy_network = QNetwork(
            observation_dim,
            action_dim,
            config.hidden_dim,
            feature_columns=feature_columns,
        ).to(self.device)
        self.target_network = QNetwork(
            observation_dim,
            action_dim,
            config.hidden_dim,
            feature_columns=feature_columns,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(config.buffer_capacity, seed=seed)
        self.action_dim = action_dim

        self.epsilon = config.epsilon_start
        self.total_steps = 0

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        """Reset strategy state between episodes. The agent itself is stateless here."""
        return None

    def select_action(self, observation: np.ndarray) -> int:
        """Return a greedy action for evaluation compatibility."""
        return self.act(observation, explore=False)

    def act(self, observation: np.ndarray, explore: bool = True) -> int:
        """Select an action using epsilon-greedy exploration."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(observation_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to replay memory."""
        self.replay_buffer.add(observation, action, reward, next_observation, done)

    def update(self) -> float | None:
        """Run one gradient update if enough replay data is available."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = self.policy_network(observations).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_network(next_observations).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_observations).gather(1, next_actions)
            targets = rewards + (1.0 - dones) * self.config.gamma * next_q_values

        loss = self.loss_fn(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    def step(self) -> None:
        """Advance internal counters and sync the target network when due."""
        self.total_steps += 1
        self.epsilon = self._current_epsilon()
        if self.total_steps % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_checkpoint(self, path: str | Path, metadata: dict[str, object]) -> Path:
        """Persist the agent state and training metadata."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "config": asdict(self.config),
                "metadata": metadata,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> dict[str, object]:
        """Load a previously saved agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        return checkpoint.get("metadata", {})

    def _current_epsilon(self) -> float:
        """Linearly decay epsilon over the configured schedule."""
        progress = min(self.total_steps / max(1, self.config.epsilon_decay_steps), 1.0)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)
