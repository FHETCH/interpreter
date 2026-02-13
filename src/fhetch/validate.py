import numpy as np

from .data import Scalar, Vector
from .fhetch_ast import Constant, Function, Return


def check_name_collision(prog):
    seen_names = set()
    for item in prog.items:
        if item.name in seen_names:
            raise ValueError("Redefinition of", item.name)
        seen_names.add(item.name)


def check_globals_types(prog, global_env):
    for item in prog.items:
        if not isinstance(item, Constant):
            continue
        if item.type == "prime":
            prime = global_env[item.name]
            if not isinstance(prime, Scalar):
                raise ValueError("Prime must be a scalar, got", prime)
        elif item.type == "primes":
            primes = global_env[item.name]
            if not isinstance(primes, Vector):
                raise ValueError("Primes must be a vector, got", primes)
            for prime in primes:
                if not isinstance(prime, (Scalar, np.uint64)):
                    raise ValueError("Primes must contain scalars, got", prime)


def check_all_funcs_return_once(prog):
    for item in prog.items:
        if not isinstance(item, Function):
            continue
        if any(isinstance(stmt, Return) for stmt in item.body[:-1]):
            raise ValueError("Multiple return statements in function", item)


def validate_all(prog, global_env):
    check_name_collision(prog)
    check_globals_types(prog, global_env)
    check_all_funcs_return_once(prog)
