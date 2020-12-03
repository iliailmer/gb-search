from typing import List

import subprocess


class GBEnvironment:
    """An Environment for computing Groebner basis
    for a given polynomial system via Maple or Magma"""

    def __init__(
        self,
        system: List[str],
        variables: List[str],
        framework: str,
        path_to_maple: str = "/Applications/Maple 2020/maple",
        path_to_magma: str = "magma",
    ) -> None:
        """
        Parameters
        ----------
        system: list of strings,
                polynomial system for which Groebner Basis is to be computed
        variables: list of strings,
                name of input variables of the system
        framework: string,
                can be either 'maple' or 'magma', chooses where to compute GB
        path_to_maple: string,
                path to Maple installation, default is
                ~/Applications/Maple 2020/maple for macOS and Maple 2020
        path_to_magma: string
                path to Magma installation
        """
        self.system = system
        self.framework = framework
        self.variables = variables
        self._path_to_framework = dict(
            maple=path_to_maple, magma=path_to_magma
        )

    def run(self):
        with open("./system.mpl", "w") as f:
            f.write(
                "sigma:=[\n\t"
                + ",\n\t".join(x for x in self.system)
                + "\n]:\n"
            )
            f.write("vars:=[" + ",".join(self.variables) + "]:\n")
            if self.framework.lower() == "maple":
                f.write(
                    "start:=time():\ngb:=Groebner"
                    "[Basis](sigma, tdeg(op(vars))):"
                    '\nfinish:=time()-start:\nwriteto("outputFile"):'
                    "\nprintf(`Time %f`, finish):"
                )
                subprocess.call(
                    (
                        f"{self._path_to_framework[self.framework]}",
                        "system.mpl",
                    ),
                    stdout=f,
                )
                return


def test():
    env = GBEnvironment(["x^2+y^10", "x^10-y"], ["x", "y"], "maple")
    env.run()
    # print(out.stdout)


if __name__ == "__main__":
    test()
