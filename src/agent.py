import os
import re
import subprocess
import time
from typing import List, Union

import numpy as np
from sympy import parse_expr
from sympy.polys import groebner
import logging

logging.getLogger().setLevel(logging.INFO)


class Agent:
    """Class Agent. It carries the substitutions for
    each variable in polynomial system"""

    def __init__(
        self,
        system: Union[str, List],
        variables: Union[str, List],
        framework: str = "maple",
        attempts: int = 1,
    ):
        """A reinforcement learning agent. This class performs Groebner Basis
        computation when the `step(substitution)` function is called.

        Parameters:
        -----------
        system : `str` or `list[str]` or `list[sympy.Poly]`.
                 An input system for which GB is to be computed.
        variables : `str` or `list[str]` or `list[sympy.Symbol]`.
                    Variables used in the provided polynomial system.
        framework : `str`.
                     The name of the framework used to compute GB.
                     Can be either `maple` or `sympy`. Default: `maple`.
        attempts : `int`.
                    The number of times GB is computed to find
                    the average runtime.
        """
        try:
            self.system = parse_expr(system)
            self.original = parse_expr(system)
            self.variables = parse_expr(variables)
        except AttributeError:
            self.system = system
            self.original = system
            self.variables = variables
        self.substitutions = [1 for _ in self.variables]
        self.framework = framework
        self.attempts = attempts

    def reset(self):
        self.system = self.original

    def substitute(self, substitutions=None):
        """apply substitutions to system"""

        substitutions = {
            v: v ** w for (v, w) in zip(self.variables, substitutions)
        }
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

    def step(self, substitutions=None, default_finish=100):
        if substitutions:
            self.substitutions = substitutions
        self.system = self.substitute(self.substitutions)
        logging.info("\nCalculating GB")
        if self.framework != "maple":
            start = time.time()
            _ = groebner(self.system, self.variables, method="f5b")
            finish = time.time() - start
        else:
            if default_finish:
                cmd = (
                    "start:=time():\ntry\n"
                    f"\tgb:=timelimit({default_finish}, "
                    f"Groebner[Basis]({self.system}, "
                    f"tdeg(op({self.variables})))):\n"
                    f'catch:\n\tprint("TIMEOUT"):\n'
                    "end try;\n"
                    "finish := time()-start;"
                )
            else:
                cmd = (
                    "start:=time();\n"
                    f"gb:=Groebner[Basis]({self.system}, "
                    f"tdeg(op({self.variables}))):\n"
                    "finish := time()-start;"
                )
            with open("tmp.mpl", "w") as f:
                f.write(cmd)
            finish = 0.0
            for attempt in range(self.attempts):
                out = os.popen("maple2020 tmp.mpl").read()
                finish += float(
                    re.findall(r"finish\s*:=\s*[-+]?[0-9]*\.?[0-9]+", out)[
                        0
                    ].split(":=")[1]
                )
            finish /= self.attempts

        logging.info(f"\tTIME: {finish}\n\tSUBSTITUTION: {self.substitutions}")
        return finish, self.substitutions

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
