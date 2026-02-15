from inspect import isfunction

import numpy as np

from .data import Vector, Scalar
from .fhetch_ast import (
    Constant,
    BinaryOperation,
    BinOp,
    MRPType,
    VecType,
    ScalarType,
    VectorLiteral,
    UnaryOperation,
    ScalarLiteral,
    VarAccess,
    VarDefinition,
    Return,
    FunctionCall,
    CallStatement,
)

MAX_U32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1
MAX_I32 = (1 << 31) - 1
MAX_I64 = (1 << 63) - 1
MIN_I32 = -(1 << 31)
MIN_I64 = -(1 << 63)


def verify_binary_op_types(lhs, rhs, op):

    l_type = "v" if isinstance(lhs, (Vector)) else "s"
    r_type = "v" if isinstance(rhs, (Vector)) else "s"

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
        # TODO: Custom exception
        raise Exception(
            f"The operation: {type(lhs).__name__} {op} {type(rhs).__name__} is not allowed"
        )


def determine_vector_type(vec: Vector) -> VecType:
    """
    Determine the VecType from a Vector instance.

    For a vector of integers: returns Vector<u32, len> (or appropriate scalar type)
    For a nested vector: recursively constructs the correct VecType
    """
    length = len(vec.value)

    if length == 0:
        # Empty vector, default to U32
        return VecType(ScalarType.U32, 0)

    # Check the first element to determine if it's nested or scalar
    first_elem = vec.value[0]

    if isinstance(first_elem, Vector):
        # Nested vector: recursively determine inner type
        inner_type = determine_vector_type(first_elem)
        return VecType(inner_type, length)
    elif isinstance(first_elem, (np.integer, int)):
        # Vector of integers: determine appropriate scalar type based on all values
        all_values = vec.value.tolist()

        # Find the largest absolute value
        largest = max(abs(max(all_values)), abs(min(all_values)))

        # Check if any value is negative
        has_negative = any(x < 0 for x in all_values)

        # Determine appropriate scalar type
        if has_negative:
            # Signed type needed
            if largest <= MAX_I32:
                inner_type = ScalarType.I32
            else:
                inner_type = ScalarType.I64
        else:
            # Unsigned type is sufficient
            if largest <= MAX_U32:
                inner_type = ScalarType.U32
            else:
                inner_type = ScalarType.U64

        return VecType(inner_type, length)
    else:
        # Handle other cases (shouldn't happen in normal flow)
        raise TypeError(f"Unexpected vector element type: {type(first_elem)}")


# TODO: can be moved inside ScalarLiteral as a method
def verify_expression_type(expr: Scalar | Vector, type: ScalarType | VecType | MRPType):
    if isinstance(type, MRPType):
        return
        # # MRP type is a vector with specific scalar type, length, and modulus
        # # First verify it as a vector with the MRP's inner type
        # vec_type = VecType(type.scalar, type.length)
        # verify_literal_type(expr, vec_type)
        # # The modulus constraint is handled separately during evaluation
        # return

    if isinstance(expr, Scalar):
        val = expr.value
        if val < 0 and type in {ScalarType.U32, ScalarType.U64}:
            raise Exception("cannot assign negative values to unsigned variables")
        if type is ScalarType.U32 and val > MAX_U32:
            raise Exception(f"{val} is out of U32 bounds")
        if type is ScalarType.U64 and val > MAX_U64:
            raise Exception(f"{val} is out of U64 bounds")

        # 32-bit Signed check
        if type is ScalarType.I32:
            if not (MIN_I32 <= val <= MAX_I32):
                raise ValueError(f"{val} is out of I32 bounds")

        # 64-bit Signed check
        if type is ScalarType.I64:
            if not (MIN_I64 <= val <= MAX_I64):
                raise ValueError(f"{val} is out of I64 bounds")
    else:
        vec_type = determine_vector_type(expr)
        if vec_type != type:
            raise ValueError(f"cannot assign {vec_type} to {type}")


def eval_expr(expr, env, global_env):
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
            lhs = eval_expr(lhs, env, global_env)
            rhs = eval_expr(rhs, env, global_env)
            verify_binary_op_types(lhs, rhs, op)
            match op:
                case BinOp.Add:
                    return lhs + rhs
                case BinOp.Sub:
                    return lhs - rhs
                case BinOp.Mul:
                    return lhs * rhs
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
            value = [eval_expr(sub_expr, env, global_env) for sub_expr in vec]
            if all(isinstance(elem, Scalar) for elem in value):
                value = np.array([elem.value for elem in value], dtype=np.uint64)
            else:
                value = np.array(value, dtype=object)
            return Vector(value)
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
        if spec.type is not None and arg.type is not None and spec.type != arg.type:
            raise ValueError(
                f"TypeError: Parameter '{spec.name}' expected type '{spec.type}', but received '{arg.type}' value: {arg}."
            )
        env[spec.name] = arg
    for statement in func.body:
        match statement:
            case VarDefinition(name, type, expr):
                res = eval_expr(expr, env, global_env)
                if type:
                    verify_expression_type(res, type)
                env[name] = res
            case CallStatement(call):
                eval_expr(call, env, global_env)
            case Return(expr):
                return eval_expr(expr, env, global_env)
            case other:
                raise NotImplementedError(other)
    return None


def eval_main(prog, global_env):
    eval_func(prog.get("main"), global_env=global_env)
