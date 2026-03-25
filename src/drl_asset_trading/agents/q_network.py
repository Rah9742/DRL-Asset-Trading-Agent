"""Q-network definition for value-based RL agents."""

from __future__ import annotations

import torch
from torch import nn

SENTIMENT_FEATURE_COLUMNS = {
    "news_count",
    "mean_overall_sentiment",
    "mean_ticker_sentiment",
    "mean_ticker_relevance",
    "weighted_ticker_sentiment",
    "sentiment_std",
    "days_since_last_news",
}


def is_sentiment_feature_name(feature_name: str) -> bool:
    """Return True when a feature belongs to the sentiment modality."""
    return feature_name in SENTIMENT_FEATURE_COLUMNS or feature_name.startswith(
        ("sentiment_mean_", "sentiment_diff_", "sentiment_window_spread_")
    )


def derive_multimodal_indices(feature_columns: list[str]) -> tuple[list[int], list[int]]:
    """Split feature columns into price-related and sentiment-related indices."""
    price_indices = [index for index, name in enumerate(feature_columns) if not is_sentiment_feature_name(name)]
    sentiment_indices = [index for index, name in enumerate(feature_columns) if is_sentiment_feature_name(name)]
    return price_indices, sentiment_indices


class QNetwork(nn.Module):
    """Estimate action values from price-only or multimodal observations."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, feature_columns: list[str]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.price_indices, self.sentiment_indices = derive_multimodal_indices(feature_columns)
        self.position_index = input_dim - 1
        self.use_multimodal_encoder = bool(self.sentiment_indices)

        if not self.use_multimodal_encoder:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            return

        price_input_dim = len(self.price_indices) + 1
        sentiment_input_dim = len(self.sentiment_indices)

        self.price_embedding = nn.Sequential(
            nn.Linear(price_input_dim, hidden_dim),
            nn.ELU(),
        )
        self.sentiment_embedding = nn.Sequential(
            nn.Linear(sentiment_input_dim, hidden_dim),
            nn.ELU(),
        )
        self.modality_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each discrete action."""
        if not self.use_multimodal_encoder:
            return self.network(inputs)

        price_inputs = torch.cat(
            (
                inputs[:, self.price_indices],
                inputs[:, self.position_index].unsqueeze(1),
            ),
            dim=1,
        )
        sentiment_inputs = inputs[:, self.sentiment_indices]
        modality_weights = torch.softmax(self.modality_logits, dim=0)
        price_embedding = self.price_embedding(price_inputs) * modality_weights[0]
        sentiment_embedding = self.sentiment_embedding(sentiment_inputs) * modality_weights[1]
        fused = torch.cat((price_embedding, sentiment_embedding), dim=1)
        return self.fusion(fused)
