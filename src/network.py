"""Neural network that generates weight vector based on the agent's reward."""

from torch import nn
import torch
from torch.nn.modules.activation import Softmax


class Network(nn.Module):
    def __init__(self, in_features: int, num_weights=6) -> None:
        super().__init__()
        self.features_to_weights = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, num_weights),
        )
        self.weights_to_log_runtime = nn.Sequential(
            nn.Tanh(), nn.Linear(num_weights, 8), nn.Tanh(), nn.Linear(8, 1)
        )

    def forward(self, x):
        logits = self.features_to_weights(x.float())
        return (logits, self.weights_to_log_runtime(logits))
