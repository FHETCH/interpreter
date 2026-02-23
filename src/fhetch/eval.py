from inspect import isfunction

import numpy as np

from .data import Vector, Scalar
from .fhetch_ast import (
    Constant,
    BinaryOperation,
    BinOp, StringLiteral,
    VectorLiteral,
    UnaryOperation,
    ScalarLiteral,
    VarAccess,
    VarDefinition,
    Return,
    FunctionCall,
    CallStatement,
)


def eval_expr(expr, env, global_env, modulo=None) -> Scalar | Vector:
    match expr:
        case ScalarLiteral(value):
            return Scalar(value)
        case VarAccess(var):
            if var in env:
                return env[var]
            elif var in global_env:
                return global_env[var]
            else:
                raise NameError("Name not defined:", var)
        case BinaryOperation(op, lhs, rhs):
            lhs = eval_expr(lhs, env, global_env, modulo)
            rhs = eval_expr(rhs, env, global_env, modulo)
            match op:
                case BinOp.Add:

                    return lhs + rhs if modulo is None else lhs.add(rhs, modulo)
                case BinOp.Sub:
                    return lhs - rhs if modulo is None else lhs.sub(rhs, modulo)
                case BinOp.Mul:
                    return lhs * rhs if modulo is None else lhs.mul(rhs, modulo)
                case BinOp.Concat:
                    assert isinstance(lhs, Vector) and isinstance(rhs, Vector)
                    return Vector(np.concatenate((lhs.value, rhs.value)))
                case BinOp.Shl:
                    assert isinstance(lhs, Scalar) and isinstance(rhs, Scalar)
                    return Scalar(lhs.value << rhs.value)
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
            value = [eval_expr(sub_expr, env, global_env, modulo) for sub_expr in vec]
            if all(isinstance(elem, Scalar) for elem in value):
                value = np.array([elem.value for elem in value], dtype=np.uint64)
            else:
                value = np.array(value, dtype=object)
            return Vector(value)
        case StringLiteral(value):
            return expr
        case other:
            raise NotImplementedError(other)


def eval_globals(prog):
    globals = {}
    for item in prog.items:
        if isinstance(item, Constant):
            globals[item.name] = eval_expr(item.value, env={}, global_env=globals)
        else:
            globals[item.name] = item
    return globals


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
                q = statement.modulus
                if q is not None:
                    q = eval_expr(q, env, global_env)
                env[name] = eval_expr(expr, env, global_env, q)
            case CallStatement(call):
                eval_expr(call, env, global_env)
            case Return(expr):
                return eval_expr(expr, env, global_env)
            case other:
                raise NotImplementedError(other)
    return None


def eval_main(prog, global_env):
    eval_func(prog.get("main"), global_env=global_env)
