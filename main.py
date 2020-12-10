import argparse
import os

from sympy import parse_expr
from torch import optim

from input_systems.generate_poly import get_system
from src.agent import Agent
from src.network import Network
from src.training import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--random", type=int, required=False, default=1)
parser.add_argument("--input-file", type=str, default="system_1")
args = parser.parse_args()

if args.random:
    s = get_system(
        "input_systems/source_systems/Cholera.mpl",
        math_symbols=None,
        rhs_size=6,
        out_size=2,
        num_y=2,
    )
    with open("system_.mpl", "w") as f:
        f.write(
            (
                "kernelopts(printbytes=false):\ninterface(echo=0,"
                + " prettyprint=0):\n"
                + 'read "input_systems/generate_poly_system.mpl":\n'
                + "sigma := [\n\t"
                + ",\n\t".join(s).replace("Derivative", "diff")
                + "\n]:\n"
                + "system_vars := GetPolySystem"
                + "(sigma, GetParameters(sigma)):\n"
                + 'writeto("system_vars.mpl"):\n'
                + "printf(`%a`, system_vars[1]);\n"
                + "printf(`\\n%a`, system_vars[2]);\n"
                + "writeto(terminal):\n"
                + "quit:"
            )
        )
    os.system("maple2020 system_.mpl")
else:
    os.system(f"maple2020 {args.input_file}.mpl")

with open("system_vars.mpl") as f:
    system_vars = [
        x.strip("[],\n").replace("^", "**").split(", ") for x in f.readlines()
    ]

system = [parse_expr(e) for e in system_vars[0]]
variables = [parse_expr(e) for e in system_vars[1]]

agent = Agent(system=system, variables=variables, attempts=3)
network = Network(in_features=1, num_weights=3)

optimizer = optim.AdamW(network.parameters(), lr=1e-3)
trainer = Trainer(
    agent=agent, network=network, optimizer=optimizer, episodes=1, epochs=100
)

if __name__ == "__main__":
    trainer.run()
