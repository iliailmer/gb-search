"""Neural network that generates weight vector based on the agent's reward."""

import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, in_features: int, num_weights=6) -> None:
        """A neural network that produces the underlying substitution sample.

        The network produces logits which are later converted by the
        trainer into the suitable substitution.
        """
        super().__init__()
        self.features_to_weights = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_weights),
        )

    def forward(self, x):
        logits = self.features_to_weights(x)
        return logits  # (logits, self.weights_to_log_runtime(logits))
