from typing import List

import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam, AdamW, Optimizer

from src.agent import Agent
from src.network import Network
import logging

logging.getLogger().setLevel(logging.INFO)
# ! terminology:
# ! the word WEIGHTS relates only to policy stuff
# ! the word SUBSTITUTIONS relates to the vector of powers for the system


class Trainer:
    def __init__(
        self,
        agent: Agent,
        network: Network,
        optimizer: Optimizer,
        episodes: int,
        epochs: int,
    ) -> None:
        """Trainer class for a reinforcement learning system.
        This is the main class responsible for sampling of substitutions
         and policy optimization.

        Parameters
        ----------
        agent : `Agent`.
                An instance of type `Agent` that runs
                the Groebner Basis computation.
        network : `Network`.
                   A PyTorch `nn.Module` that produces
                   logits for the substitution sampling.
        optimizer : `torch.Optimizer`.
                     A PyTorch optimizer instance.
        episodes : `int`.
                    Number of episodes to run the sequence
                    of sampling and GB computations.
        epoch : `int`.
                 Number of training epochs.
        """
        self.agent = agent
        self.network = network
        self.optimizer = optimizer
        self.trajectory: List = list()
        self.episodes = episodes
        self.runtimes: List[float] = list()
        self.substitutions: List[List[int]] = list()
        self.epochs = epochs
        self.losses: List[float] = []

    def loss_fn(
        self,
        runtimes: torch.Tensor,
        substitutions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        runtime: observations
        substitutions: actions
        weights: total reward per batch
        """
        logits = self.network(runtimes)  # network takes in obersvation
        # create a local sampler based on logits
        sampler = Categorical(logits=logits)
        # create log probability from actions
        logp = sampler.log_prob(substitutions)
        return -(logp * weights).mean()

    def run(self):
        for epoch in range(self.epochs):

            self.runtimes = []
            self.substitutions = []

            # generate all-1 substitution
            # if this is the first epoch use all 1
            # otherwise use a random state
            # this is equivalent to a reset
            if epoch == 0:
                substitutions = [1 for _ in self.agent.variables]
            else:
                substitutions = [
                    max(self.sampler.sample().item(), 1)
                    for _ in self.agent.variables  # noqa
                    # logits are defined for non-0 epoch
                ]

            for _ in range(self.episodes):
                # begin trajectory/batch
                # act in the environment
                runtime, substitutions = self.agent.step(
                    substitutions, default_finish=500
                )
                # save the result of this experiment
                self.runtimes.append(runtime)
                self.substitutions.append(substitutions)

                # recall: runtime = observation
                # use observation to generate new logits
                runtime = torch.tensor([runtime]).view((-1, 1))
                logits = self.network(runtime)
                # use new logits to generate new action (aka substitution)
                self.sampler = Categorical(logits=logits)
                substitutions = [
                    max(self.sampler.sample().item(), 1)
                    for _ in self.agent.variables
                ]
            self.weights = torch.tensor(self.runtimes)
            self.optimizer.zero_grad()
            loss = self.loss_fn(
                torch.tensor(self.runtimes).view((-1, 1)),
                torch.tensor(self.substitutions).T,
                self.weights,
            )
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            logging.info(f"EPOCH: {epoch}, LOSS: {loss.item()}")
