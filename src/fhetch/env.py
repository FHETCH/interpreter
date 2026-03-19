import sys

import numpy as np
from .data import Scalar, Vector, MRP
from .fhetch_ast import Constant
from .ntt import ROOTS_UNITY

# TODO: import from client to interp seems a bit off
# Import serialization utilities for MRP I/O
from client import serialization


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


def builtin_get(vector: Vector, index: Scalar):
    return vector.value[index.value]


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

def builtin_automorph_eval(lhs: Vector, rot: Scalar):
    assert isinstance(lhs, Vector)
    degree = len(lhs.value)
    exp = pow(5,rot.value,2*degree)
    return lhs.automorph_eval(exp)


def builtin_read_mrp_u32_Q(path: str, Q: Vector):
    """Load a single MRP from disk for the Q modulus set.

    Args:
        path: Path to the .npz file containing the MRP

    Returns:
        The loaded MRP object
    """
    mrp = serialization.load_mrp(path)
    assert mrp.base() == set(x for x in Q.value)
    return mrp


def builtin_write_mrp(mrp, path: str):
    """Save a single MRP to disk for the Q modulus set.

    Args:
        mrp: The MRP object to save
        path: Output file path (will be created/overwritten)
    """
    serialization.save_mrp(mrp, path)


def builtin_base_extend(mrp: MRP, digit_base: Vector, full_base: Vector):
    digit_base_set = set(int(x) for x in digit_base.value)
    full_base_set = set(int(x) for x in full_base.value)
    new_primes_set = full_base_set - digit_base_set
    return mrp.extract_base(digit_base_set).extend_base(new_primes_set, True)


def builtin_rescale(mrp: MRP, moduli):
    q = set(int(x) for x in moduli)
    rescaled_q = mrp.divq(q)
    return rescaled_q

def builtin_rotate(mrp: MRP, rotation:Scalar):
    return mrp.automorph(rotation.value)


def builtin_get_limb(mrp: MRP, prime: Scalar):
    return mrp.values[prime.value]


def builtin_set_limb(mrp: MRP, prime: Scalar, vec: Vector):
    new_values = dict(mrp.values)
    new_values[prime.value] = vec
    return MRP(new_values)


def builtin_empty_mrp():
    return MRP({})


def default_global():
    return {
        "write": builtin_write,
        "read_mrp_u32_Q": builtin_read_mrp_u32_Q,
        "write_mrp_u32": builtin_write_mrp,
        "print": builtin_print,
        "get": builtin_get,
        "sr_addp": builtin_add,
        "sr_subp": builtin_sub,
        "sr_mulp": builtin_mul,
        "sr_set_rou": builtin_set_rou,
        "sr_NTT": builtin_ntt,
        "sr_iNTT": builtin_intt,
        "sr_automorph_eval": builtin_automorph_eval,
        "BaseExtend": builtin_base_extend,
        "Rescale": builtin_rescale,
        "Rotate": builtin_rotate,
        "get_mrp_limb": builtin_get_limb,
        "set_mrp_limb": builtin_set_limb,
        "empty_mrp": builtin_empty_mrp,
    }
