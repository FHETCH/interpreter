import sys

from interp.fhetch_ast import Constant, ScalarLiteral, VectorLiteral

def builtin_write(obj):
    match obj:
        case ScalarLiteral(x):
            sys.stdout.buffer.write(x.to_bytes(4, "little"))
        case VectorLiteral(v):
            for x in v:
                builtin_write(x)
        case Constant(_, _, value):
            builtin_write(value)
        case other:
            print("Unknown type", type(other), file=sys.stderr)
            print(other)


def builtin_add(lhs, rhs, q):
    assert isinstance(lhs, VectorLiteral) and isinstance(rhs, VectorLiteral)
    assert len(lhs.value) == len(rhs.value)
    return (lhs + rhs) % q


def builtin_sub(lhs, rhs, q):
    assert isinstance(lhs, VectorLiteral) and isinstance(rhs, VectorLiteral)
    assert len(lhs.value) == len(rhs.value)
    return (lhs - rhs) % q


def builtin_mul(lhs, rhs, q):
    if isinstance(rhs, ScalarLiteral):
        return lhs.mmuls(rhs, q)
    elif isinstance(rhs, VectorLiteral):
        return lhs.mmulv(rhs, q)
    # integers??
    return (lhs * rhs) % q


def default_global():
    return {
        "write": builtin_write,
        "sr_addp": builtin_add,
        "sr_subp": builtin_sub,
        "sr_mulp": builtin_mul,
    }
