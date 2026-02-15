from inspect import isfunction

from interp.env import default_global
from interp.fhetch_ast import (
    Constant,
    BinaryOperation,
    BinOp,
    ScalarType,
    VecType,
    MRPType,
    VectorLiteral,
    UnaryOperation,
    ScalarLiteral,
    VarAccess,
    VarDefinition,
    Return,
    FunctionCall,
    CallStatement,
    Function,
)
from interp.parser import VectorType


MAX_U32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1
MAX_I32 = (1 << 31) - 1
MAX_I64 = (1 << 63) - 1
MIN_I32 = -(1 << 31)
MIN_I64 = -(1 << 63)


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
        # TODO: Custom exception
        raise Exception(
            f"The operation: {type(lhs).__name__} {op} {type(rhs).__name__} is not allowed"
        )


# TODO: can be moved inside ScalarLiteral as a method
def verify_literal_type(expr: ScalarLiteral|VectorLiteral, type: ScalarType|VecType|MRPType):
    if isinstance(type, MRPType):
        return
        # # MRP type is a vector with specific scalar type, length, and modulus
        # # First verify it as a vector with the MRP's inner type
        # vec_type = VecType(type.scalar, type.length)
        # verify_literal_type(expr, vec_type)
        # # The modulus constraint is handled separately during evaluation
        # return
    
    if isinstance(expr, ScalarLiteral):
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
        for element in expr.value:
            verify_literal_type(element, type.inner)
            element.type = type.inner
                
        
    


def determine_literal_type(expr: ScalarLiteral | VectorLiteral) -> ScalarType | VecType:
    if isinstance(expr, ScalarLiteral):
        val = expr.value
        if val < 0:
            return ScalarType.I32 if MIN_I32 <= val <= MAX_I32 else ScalarType.I64
        return ScalarType.U32 if val <= MAX_U32 else ScalarType.U64
    
    # For VectorLiteral, recursively determine inner type
    vec = expr.value
    if len(vec) == 0:
        # Empty vector, default to U32
        return VecType(ScalarType.U32, 0)
    
    # Check if elements are nested vectors or scalars
    first_element = vec[0]
    if isinstance(first_element, VectorLiteral):
        # Nested vector: recursively determine inner type
        inner_type = determine_literal_type(first_element)
        return VecType(inner_type, len(vec))
    elif isinstance(first_element, ScalarLiteral):
        # Vector of scalars: determine scalar type based on all values
        all_values = [elem.value for elem in vec]
        largest = max(abs(max(all_values)), abs(min(all_values)))
        if any(x < 0 for x in all_values):
            inner_type = ScalarType.I32 if largest <= MAX_I32 else ScalarType.I64
        else:
            inner_type = ScalarType.U32 if largest <= MAX_U32 else ScalarType.U64
        return VecType(inner_type, len(vec))
    else:
        # Fallback for plain int values (shouldn't happen in normal flow)
        all_values = [elem if isinstance(elem, int) else elem.value for elem in vec]
        largest = max(abs(max(all_values)), abs(min(all_values)))
        if any(x < 0 for x in all_values):
            inner_type = ScalarType.I32 if largest <= MAX_I32 else ScalarType.I64
        else:
            inner_type = ScalarType.U32 if largest <= MAX_U32 else ScalarType.U64
        return VecType(inner_type, len(vec))


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
            verify_binary_op_types(lhs, rhs, op)
            match op:
                case BinOp.Add:
                    return lhs + rhs
                case BinOp.Sub:
                    return lhs - rhs
                case BinOp.Mul:
                    return lhs * rhs
                case BinOp.Concat:
                    assert isinstance(lhs, VectorLiteral) and isinstance(
                        rhs, VectorLiteral
                    )
                    return VectorLiteral(lhs.value + rhs.value)
                case BinOp.Shl:
                    assert isinstance(lhs, ScalarLiteral) and isinstance(
                        rhs, ScalarLiteral
                    )
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
            return VectorLiteral(
                [eval_expr(sub_expr, env, global_env) for sub_expr in vec]
            )
        case other:
            raise NotImplementedError(other)


def eval_consts(prog):
    consts = {}
    for item in prog.items:
        if isinstance(item, Constant):
            consts[item.name] = item.value = eval_expr(
                item.value, env={}, global_env=consts
            )


def eval_func(func, *args, global_env):
    if isfunction(func):
        return func(*args)
    func: Function = func
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
                    verify_literal_type(res, type)
                else:
                    type = determine_literal_type(res)
                res.type = type
                env[name] = res
            case CallStatement(call):
                eval_expr(call, env, global_env)
            case Return(expr):
                return eval_expr(expr, env, global_env)
            case other:
                raise NotImplementedError(other)
    return None


def eval_main(prog):
    global_env = default_global()
    global_env.update(
        {
            item.name: item if isinstance(item, Function) else item.value
            for item in prog.items
        }
    )
    eval_func(prog.get("main"), global_env=global_env)
