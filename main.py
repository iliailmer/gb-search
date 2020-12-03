from input_systems.utils import get_example_2
from torch import optim
from torch.nn import L1Loss

from input_systems import get_example_2
from src.agent import Agent
from src.network import Network
from src.training import Trainer
from src.generate_poly import generate_rhs
from sympy import symbols, parse_expr, simplify


def generate_system(size: int = 5, num_variables: int = 5):
    variables = symbols(f"x1:{num_variables}")
    system = []
    i = 0
    x = simplify(
        parse_expr(generate_rhs(5, params=variables, functions=variables))
    )
    while i < len(variables):
        if x != 0:
            system.append(x)
            x = simplify(
                parse_expr(
                    generate_rhs(5, params=variables, functions=variables)
                )
            )
            i += 1
        else:
            x = simplify(
                parse_expr(
                    generate_rhs(5, params=variables, functions=variables)
                )
            )
    return variables, system


(
    variables,
    system,
) = get_example_2()  # generate_system(size=5, num_variables=5)
agent = Agent(system, variables)
network = Network(
    in_features=len(agent.variables), num_weights=len(agent.variables)
)
loss_fn = None  # L1Loss()
optimizer = optim.AdamW(network.parameters(), lr=1e-3)
trainer = Trainer(agent, network, optimizer, loss_fn=loss_fn, epochs=10)

if __name__ == "__main__":
    trainer.run()
