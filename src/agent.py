import re
import subprocess
from typing import List
from sympy.polys import groebner
import time
import numpy as np
from sympy import parse_expr
from typing import Union


class Agent:
    """Class Agent. It carries the weights for
    each variable in polynomial system"""

    def __init__(
        self,
        system: Union[str, List],
        variables: Union[str, List],
        framework: str = "maple",
    ):
        try:
            self.system = parse_expr(system)
            self.original = parse_expr(system)
            self.variables = parse_expr(variables)
        except AttributeError:
            self.system = system
            self.original = system
            self.variables = variables
        self.weights = [1 for _ in self.variables]
        self.framework = framework
        self._path_to_framework = dict(maple="/Applications/Maple 2020/maple")

    def substitute(self, weights=None):
        """apply weights to system"""

        substitutions = {v: v ** w for (v, w) in zip(self.variables, weights)}
        self.system = [s.subs(substitutions) for s in self.original]
        return self.system

    def __repr__(self) -> str:
        return (
            "Agent\n"
            + "\tsystem:\n\t\t"
            + ", \n\t\t".join([str(x) for x in self.system])
            + "\n\tvariables:\n\t\t"
            + ", ".join([str(x) for x in self.variables])
        )

    def step(self, weights=None):
        if weights:
            self.weights = weights
        self.system = self.substitute(self.weights)
        print("Calculating GB")
        start = time.time()
        _ = groebner(self.system, self.variables, method="f5b")
        finish = time.time() - start
        reward = -np.log(
            finish + 1e-8  # eps, just in case
        )  # if we maximize reward, we want to minimize time
        return reward, self.weights

        # with open("./src/system.mpl", "w") as f:
        #     f.write(
        #         "sigma:=[\n\t"
        #         + ",\n\t".join([str(x) for x in self.system])
        #         + "\n]:\n"
        #     )
        #     f.write(
        #         "vars:=[" + ",".join(str(x) for x in self.variables) + "]:\n"
        #     )
        #     f.write(
        #         "start:=time():\ngb:=Groebner"
        #         "[Basis](sigma, tdeg(op(vars))):"
        #         '\nfinish:=time()-start:\nwriteto("outputFile"):'
        #         "\nprintf(`Time %f`, finish):"
        #     )

        # if self.framework.lower() == "maple":
        #     dump = open("dump.txt", "w")
        #     subprocess.call(
        #         [
        #             # f"{self._path_to_framework[self.framework]}",
        #             "/Applications/Maple\ 2020/maple "
        #             "src/system.mpl",
        #         ],
        #         shell=True
        #         # stdout=dump,
        #     )
        #     dump.close()
        # with open("src/outputFile", "r") as out_file:
        #     time = float(
        #         re.search(r"Time ([0-9.]+)", out_file.read()).group(1)
        #     )