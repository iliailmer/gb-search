import argparse
import re
from typing import List, Tuple, Union

import numpy as np
from sympy import Add, Derivative, Function, Mul, Symbol, parse_expr, symbols

Expression = Union[Add, Mul]
Symbols = List[Symbol]
Functions = List[Function]
Derivatives = List[Derivative]
__Symbols = Union[Expression, Symbols, Tuple[str, ...]]


def read_system(name: str) -> Tuple[str, ...]:
    with open(name, "r") as f:
        file = f.read()
    sigma = (
        re.findall(r"sigma\s*:=\s*.*(?=y1)", file, re.DOTALL)[0]
        .replace("subs(known_data, ", "")
        .split(":=")[1]
        .strip("[],\t\n ")
        .replace("=", "-")
    )
    y_outputs = (
        re.findall(r"y1.*]", file, re.DOTALL)[0]
        .strip("[],\t\n ")
        .replace("=", "-")
    )
    return sigma, y_outputs


def get_right_hand_sides(sigma: str) -> List[Expression]:
    expressions = [parse_expr(eq) for eq in list(sigma.split(",\n"))]
    return expressions


def get_params(expression: Expression) -> Symbols:
    return [each for each in (expression.atoms(Symbol)) if each != Symbol("t")]


def get_functions(expression: Expression) -> Functions:
    return list(expression.atoms(Function))


def get_derivatives(expression: Expression) -> Derivatives:
    return list(expression.atoms(Derivative))


def generate_rhs(
    size: int,
    math_symbols: __Symbols = ("+", "-", "*"),
    params: __Symbols = symbols("a:e"),
    functions: __Symbols = (Function("x(t)"), Function("y(t)")),
):
    size = np.random.randint(size // 2, size)
    out = list()
    for _ in range(size):
        try:
            param = str(params[np.random.choice(len(params), replace=True)])
        except (IndexError, ValueError):
            param = ""
        sym = str(
            math_symbols[np.random.choice(len(math_symbols), replace=True)]
        )
        try:
            fun = str(
                functions[np.random.choice(len(functions), replace=True)]
            )
        except (IndexError, ValueError):
            fun = ""
        if param or fun:
            out.append(f"({param}{sym}{fun})")
        else:
            out.append("")
    result = ""
    for each in out:
        result = (
            result
            + each
            + [" + ", " - ", " * "][np.random.choice(3, p=[0.4, 0.4, 0.2])]
        )
    return result.strip("+-* ")


parser = argparse.ArgumentParser(
    "RandomODE",
    description="""
    Generate Random ODE system based on a source system.
    contained in a .mpl (or any other) file.
    System must be detectable via regex r'sigma\\s*:=\\s*.*]:',
    e.g. `sigma:= [diff(x(t), t)=x]:`
    """,
)

parser.add_argument(
    "-n", "--name", type=str, required=True, help="Name of the source file"
)
parser.add_argument(
    "-o",
    "--output-name",
    type=str,
    required=True,
    help="Name of the output file",
)
parser.add_argument(
    "-s",
    "--rhs-size",
    type=int,
    required=False,
    help="Number of terms in the output rhs for ODEs",
    default=4,
)
parser.add_argument(
    "-os",
    "--out-size",
    type=int,
    required=False,
    help="Number of terms in the output rhs for y-functions",
    default=3,
)
parser.add_argument(
    "-ny",
    "--num-y",
    type=int,
    required=False,
    help="Number of y-functions (at most 1)",
    default=2,
)
parser.add_argument(
    "-ms",
    "--math-symbols",
    nargs="+",
    help="(optional) List of math operations, i.e. +, -, etc.",
    required=False,
    default="",
)

if __name__ == "__main__":
    args = parser.parse_args()
    result = []
    sigma, y_outputs = read_system(args.name)
    expressions = get_right_hand_sides(sigma)
    y_expressions = get_right_hand_sides(y_outputs)
    params = []
    functions = []
    derivatives = []
    y_functions = []
    for idx, each in enumerate(expressions):
        params.extend(get_params(each))
        functions.extend(get_functions(each))
        derivatives.extend(get_derivatives(each))
    for idx, each in enumerate(y_expressions):
        fun = get_functions(each)
        for other in fun:
            if other not in functions:
                y_functions.append(other)
    functions = list(set(functions))
    params = list(set(params))
    params.append("1")
    derivatives = list(set(derivatives))
    print("Functions: ", functions)
    print("Output functions (y_i(t)): ", y_functions)
    print("Parameters: ", params)
    print("Derivatives: ", derivatives)
    if not args.math_symbols:
        ms = [" * ", " + ", " - "]
    else:
        ms = args.math_symbols
    for idx, each in enumerate(expressions):
        generated = generate_rhs(args.rhs_size, ms, params, functions)
        if derivatives:
            if str(derivatives[idx % len(derivatives)].args[0]) in generated:
                result.append(
                    (
                        f"{derivatives[idx % len(derivatives)]}"
                        + " = "
                        + f"{generated}"
                    )
                )
                # print(
                #     derivatives[idx % len(derivatives)],
                #     " did not require forcing of unknown function into rhs",
                # )
            else:
                lhs_functions = (
                    str(params[np.random.choice(len(params))])
                    + str(ms[np.random.choice(len(ms))])
                    + str(derivatives[idx % len(derivatives)].args[0])
                )
                result.append(
                    (
                        f"{derivatives[idx % len(derivatives)]}"
                        + " = "
                        + f"{generated}"
                        + [" + ", " - ", " * "][np.random.choice(3)]
                        + lhs_functions
                    )
                )
        else:
            result.append(("0=" + f"{generated}"))
    if args.num_y < 0:
        for idx, each in enumerate(y_expressions):
            generated = generate_rhs(args.out_size, ms, params, functions)
            if derivatives:
                result.append((f"{y_functions[idx]}" + " = " + f"{generated}"))
            else:
                result.append(("0=" + f"{generated}"))
    else:
        for idx in range(args.num_y):
            generated = generate_rhs(args.out_size, ms, params, functions)
            if derivatives:
                result.append((f"y{idx+1}(t)" + " = " + f"{generated}"))
            else:
                result.append(("0=" + f"{generated}"))
    with open(args.output_name, "w") as f:
        if "_" in args.output_name:
            name = args.output_name.split("_")[1].split(".")[0]
        else:
            name = args.output_name
        f.write(
            (
                "kernelopts(printbytes=false):\n"
                + 'read "generate_poly_system.mpl":\n'
                + 'read "get_stats_from_polynomials.mpl":\n'
                + "sigma := [\n\t"
                + ",\n\t".join(result).replace("Derivative", "diff")
                + "\n]:\n"
                + "system_vars := GetPolySystem"
                + "(sigma, GetParameters(sigma)):\n"
                + "GetStats(system_vars[1], system_vars[2]):\n"
            )
        )
