from interp.fhetch_ast import Constant, ScalarLiteral, VectorLiteral, Function, Return


def check_name_collision(prog):
    seen_names = set()
    for item in prog.items:
        if item.name in seen_names:
            raise ValueError("Redefinition of", item.name)
        seen_names.add(item.name)


def check_globals_types(prog):
    for item in prog.items:
        if not isinstance(item, Constant):
            continue
        if item.type == "prime" and not isinstance(item.value, ScalarLiteral):
            raise ValueError("Prime must be a scalar, got", item.value)
        elif item.type == "primes":
            if not isinstance(item.value, VectorLiteral):
                raise ValueError("Primes must be a vector, got", item.value)
            for expr in item.value:
                if not isinstance(expr, ScalarLiteral):
                    raise ValueError("Primes must contain scalars, got", expr)


def check_all_funcs_return_once(prog):
    for item in prog.items:
        if not isinstance(item, Function):
            continue
        if any(isinstance(stmt, Return) for stmt in item.body[:-1]):
            raise ValueError("Multiple return statements in function", item)


def validate_all(prog):
    check_name_collision(prog)
    check_globals_types(prog)
    check_all_funcs_return_once(prog)
