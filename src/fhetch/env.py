import sys

import numpy as np

from .data import Scalar, Vector
from .fhetch_ast import Constant, ScalarLiteral
from .ntt import ROOTS_UNITY


def builtin_write(obj):
    match obj:
        case Scalar(x):
            sys.stdout.buffer.write(int(x).to_bytes(4, "little"))
        case np.uint64():
            sys.stdout.buffer.write(obj.tobytes())
        case Vector(v):
            if v.dtype.kind in ('u', 'i'):
                # optimization for vectors of scalars
                sys.stdout.buffer.write(v.astype(np.int32).tobytes())
            else:
                for x in v:
                    builtin_write(x)
        case Constant(_, _, value):
            builtin_write(value)
        case other:
            print("Unknown type", type(other), file=sys.stderr)
            print(other)


def builtin_print(obj):
    match obj:
        case Scalar(x):
            print(x, end='')
        case Vector(v):
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
    assert isinstance(lhs, Vector) and isinstance(rhs, Vector)
    assert len(lhs.value) == len(rhs.value)
    return lhs.add(rhs, q.value)


def builtin_sub(lhs, rhs, q):
    assert isinstance(lhs, Vector) and isinstance(rhs, Vector)
    assert len(lhs.value) == len(rhs.value)
    return lhs.sub(rhs, q.value)


def builtin_mul(lhs, rhs, q):
    return lhs.mul(rhs, q.value)

def builtin_set_rou(ring_dimension: Scalar, modulus: Scalar, rou: Scalar):
    """Set the global root of unity for this modulus"""
    assert ring_dimension.value > 0
    ROOTS_UNITY[(ring_dimension.value, modulus.value)] = rou.value
    return

def builtin_ntt(lhs: Vector, q: Scalar):
    assert isinstance(lhs, Vector)
    return lhs.forward_ntt(q, rou=ROOTS_UNITY.get((len(lhs.value), q.value)))

def builtin_intt(lhs: Vector, q: Scalar):
    assert isinstance(lhs, Vector)
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
