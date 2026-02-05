from inspect import isfunction

from interp.env import default_global
from interp.fhetch_ast import Constant, BinaryOperation, BinOp, VectorLiteral, UnaryOperation, ScalarLiteral, VarAccess, \
    VarDefinition, Return, FunctionCall, CallStatement, Function


def eval_expr(expr, env, global_env):
    match expr:
        case ScalarLiteral(_):
            return expr
        case VarAccess(var):
            if var in env:
                return env[var]
            elif var in global_env:
                return global_env[var]
            else:
                raise NameError("Name not defined:", var)
        case BinaryOperation(op, lhs, rhs):
            lhs = eval_expr(lhs, env, global_env)
            rhs = eval_expr(rhs, env, global_env)
            verify_binary_op_types(lhs,rhs,op)
            match op:
                case BinOp.Add:
                    return lhs + rhs
                case BinOp.Sub:
                    return lhs - rhs
                case BinOp.Mul:
                    return lhs * rhs
                case BinOp.Concat:
                    assert isinstance(lhs, VectorLiteral) and isinstance(rhs, VectorLiteral)
                    return VectorLiteral(lhs.value + rhs.value)
                case BinOp.Shl:
                    assert isinstance(lhs, ScalarLiteral) and isinstance(rhs, ScalarLiteral)
                    return ScalarLiteral(lhs.value << rhs.value)
                case other:
                    raise NotImplementedError(other)
        case UnaryOperation(operation, operand):
            operand = eval_expr(operand, env, global_env)
            if operation == "-":
                return -operand
            else:
                raise ValueError(operation)
        case FunctionCall(name, args):
            func = global_env[name]
            args = [eval_expr(expr, env, global_env) for expr in args]
            return eval_func(func, *args, global_env=global_env)
        case VectorLiteral(vec):
            return VectorLiteral([eval_expr(sub_expr, env, global_env) for sub_expr in vec])
        case other:
            raise NotImplementedError(other)

def verify_binary_op_types(lhs, rhs, op):

    l_type = "v" if isinstance(lhs, (VectorLiteral)) else "s"
    r_type = "v" if isinstance(rhs, (VectorLiteral)) else "s"

    # 2. Define the "Allow List"
    # Format: {operator: {(lhs_type, rhs_type), ...}}
    allowed_ops = {
        BinOp.Add: {("s", "s"), ("v", "v")},
        BinOp.Sub: {("s", "s"), ("v", "v")},
        BinOp.Mul: {("s", "s"), ("v", "s"), ("s", "v"), ("v", "v")},
        BinOp.Concat: {("v", "v")},
        BinOp.Shl: {("s", "s")},
        BinOp.Shr: {("s", "s")},
    }

    if (l_type, r_type) not in allowed_ops[op]:
        #TODO: Custom exception
        raise Exception(f"The operation: {type(lhs).__name__} {op} {type(rhs).__name__} is not allowed")
    
def eval_consts(prog):
    consts = {}
    for item in prog.items:
        if isinstance(item, Constant):
            consts[item.name] = item.value = eval_expr(item.value, env={}, global_env=consts)


def eval_func(func, *args, global_env):
    if isfunction(func):
        return func(*args)
    env = global_env.copy()
    # update env with arguments
    for spec, arg in zip(func.args, args):
        # TODO: type checking
        env[spec.name] = arg
    for statement in func.body:
        match statement:
            case VarDefinition(name, _type, expr):
                env[name] = eval_expr(expr, env, global_env)
            case CallStatement(call):
                eval_expr(call, env, global_env)
            case Return(expr):
                return eval_expr(expr, env, global_env)
            case other:
                raise NotImplementedError(other)
    return None


def eval_main(prog):
    global_env = default_global()
    global_env.update({item.name: item if isinstance(item, Function) else item.value for item in prog.items})
    eval_func(prog.get("main"), global_env=global_env)
