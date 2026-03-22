"""Replay buffer used by value-based RL agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(slots=True)
class TransitionBatch:
    """Mini-batch sampled from replay memory."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """A simple FIFO replay buffer for off-policy learning."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = capacity
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self._buffer.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random mini-batch from memory."""
        transitions = self._random.sample(self._buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*transitions)
        return TransitionBatch(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.asarray(next_observations, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )
