import sys

from interp.fhetch_ast import Constant, ScalarLiteral, VectorLiteral
from numpy import argsort, array

# Root of unity used for the NTT (if None, use the default from Sympy)
ROOTS_UNITY = {}

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


def builtin_print(obj):
    match obj:
        case ScalarLiteral(x):
            print(x, end='')
        case VectorLiteral(v):
            print('[', end='')
            for x in v:
                builtin_print(x)
                print(', ', end='')
            print(']')
        case Constant(_, _, value):
            builtin_print(value)
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

def builtin_set_rou(ring_dimension: ScalarLiteral, modulus: ScalarLiteral, rou: ScalarLiteral):
    """Set the global root of unity for this modulus"""
    assert ring_dimension.value > 0
    ROOTS_UNITY[(ring_dimension.value, modulus.value)] = rou.value
    return

def builtin_ntt(lhs: VectorLiteral, q: ScalarLiteral):
    assert isinstance(lhs, VectorLiteral)
    return lhs.forward_ntt(q, rou=ROOTS_UNITY.get((len(lhs.value), q.value)))

def builtin_intt(lhs: VectorLiteral, q: ScalarLiteral):
    assert isinstance(lhs, VectorLiteral)
    return lhs.inverse_ntt(q, rou=ROOTS_UNITY.get((len(lhs.value), q.value)))


def default_global():
    return {
        "write": builtin_write,
        "print": builtin_print,
        "sr_addp": builtin_add,
        "sr_subp": builtin_sub,
        "sr_mulp": builtin_mul,
        "sr_set_rou": builtin_set_rou,
        "sr_NTT": builtin_ntt,
        "sr_iNTT": builtin_intt,
    }
