from typing import List

import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam, AdamW, Optimizer

from src.agent import Agent
from src.network import Network


class Trainer:
    def __init__(
        self,
        agent: Agent,
        network: Network,
        optimizer: Optimizer,
        loss_fn,
        epochs: int,
    ) -> None:
        """Initialize trainer.
        Parameters
        ----------

        """
        self.agent = agent
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.trajectory: List = list()
        self.epochs = epochs

    def step(
        self,
        system_features: torch.Tensor,
    ):
        # generate weights
        logits, log_rt_estimate = self.network(system_features)
        self.sampler = Categorical(logits=logits)
        weights = [self.sampler.sample().item() + 1 for _ in logits]
        self.trajectory.append(self.agent.step(weights))
        loss = self.loss_fn(
            log_rt_estimate, torch.tensor([self.trajectory[-1][0]])
        )
        return loss, weights

    def run(self):
        starting_features = torch.tensor([1 for _ in self.agent.variables])
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss, weights = self.step(starting_features)
            starting_features = torch.tensor(weights)
            loss.backward()
            self.optimizer.step()
            print(f"EPOCH: {epoch}, LOSS: {loss.item()}")

    def _loss_fn(self, observation, action, weights):
        logp = get_policy(observation).log_prob(action)
        return -(logp * weights).mean()
