from .fhetch_ast import Function, VarDefinition, VarAccess, ScalarLiteral, BinaryOperation, VectorLiteral, \
    FunctionCall, BinOp


def desugar(prog):
    for item in prog.items:
        if isinstance(item, Function):
            _desugar_func(item)


def _desugar_func(func):
    for instr in func.body:
        if isinstance(instr, VarDefinition) and instr.modulus is not None:
            instr.definition = _rewrite_expr(instr.definition, instr.modulus)
            instr.modulus = None

def _rewrite_expr(expr, modulus):
    match expr:
        case VarAccess(_) | ScalarLiteral(_):
            return expr
        case BinaryOperation(op, lhs, rhs):
            new_lhs = _rewrite_expr(lhs, modulus)
            new_rhs = _rewrite_expr(rhs, modulus)
            match op:
                case BinOp.Add:
                    return FunctionCall("sr_addp", [new_lhs, new_rhs, modulus])
                case BinOp.Sub:
                    return FunctionCall("sr_subp", [new_lhs, new_rhs, modulus])
                case BinOp.Mul:
                    return FunctionCall("sr_mulp", [new_lhs, new_rhs, modulus])
            return BinaryOperation(op, new_lhs, new_rhs)
        case VectorLiteral(values):
            new_values = [_rewrite_expr(v, modulus) for v in values]
            if all(a is b for a, b in zip(values, new_values)):
                return expr
            return VectorLiteral(new_values)
        case other:
            raise NotImplementedError(other)
