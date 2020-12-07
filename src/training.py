from typing import List

import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam, AdamW, Optimizer

from src.agent import Agent
from src.network import Network

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
        """Initialize trainer.
        Parameters
        ----------

        """
        self.agent = agent
        self.network = network
        self.optimizer = optimizer
        self.trajectory: List = list()
        self.episodes = episodes
        self.runtimes = list()
        self.rewards = list()
        self.substitutions = list()
        self.epochs = epochs

    def loss_fn(self, runtimes, substitutions, weights):
        """
        runtime: observations
        substitutions: actions
        weights: total reward per batch (trajectory)
        """
        logits = self.network(runtimes)  # network takes in obersvation
        self.sampler = Categorical(
            logits=logits
        )  # create sampler based on logits
        logp = self.sampler.log_prob(
            substitutions
        )  # create log probability from actions
        return -(logp * weights).mean()

    def run(self):
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # generate all-1 substitution
            # if this is the first epoch use all 1
            # otherwise use a random state
            # this is equivalent to a reset
            if epoch == 0:
                substitutions = [1 for _ in self.agent.variables]
            else:
                substitutions = torch.randint(
                    1,
                    len(self.agent.variables),
                    (len(self.agent.variables),),
                ).tolist()

            for _ in range(self.episodes):
                # begin trajectory/batch

                # act in the environment
                runtime, substitutions = self.agent.step(
                    substitutions, default_finish=100
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
                    for _ in range(logits.shape[1])
                ]

            total_reward = sum(self.runtimes)
            episode_length = len(self.runtimes)  # 10

            weights = torch.tensor([[-total_reward] * episode_length])
            loss = self.loss_fn(
                torch.tensor(self.runtimes).view((-1, 1)),
                torch.tensor(self.substitutions).view((-1, episode_length)),
                weights,
            )
            loss.backward()
            self.optimizer.step()
            print(f"EPOCH: {epoch}, LOSS: {loss.item()}")
