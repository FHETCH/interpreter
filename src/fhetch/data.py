"""Data model for FHETCH programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy import typing as npt

from .ntt import _nb_theory_scratchpad, _number_theoretic_transform


def modulo(x, q):
    x %= q
    if x.dtype == np.int64:
        x -= (x > q//2) * q
    return x


@dataclass(frozen=True)
class Scalar:
    # We use Python's arbitrary precision int for scalars in order to allow
    # arbitrary computations to produce primes.
    value: int

    def __add__(self, other):
        return Scalar(self.value + other.value)

    def __sub__(self, other):
        return Scalar(self.value - other.value)

    def __mul__(self, other):
        return Scalar(self.value * other.value)

    def __mod__(self, q):
        return Scalar(self.value % q.value)

    def add(self, other, q):
        return Scalar(modulo(self.value + other.value, q.value))

    def sub(self, other, q):
        return Scalar(modulo(self.value - other.value, q.value))

    def mul(self, other, q):
        return Scalar(modulo(self.value * other.value, q.value))


@dataclass
class Vector:
    value: np.array

    def __add__(self, other):
        return Vector(self.value + other.value)

    def __sub__(self, other):
        return Vector(self.value - other.value)

    def __mul__(self, other):
        return Vector(self.value * other.value)

    def __mod__(self, other):
        if isinstance(other, Scalar):
            other = other.value
        return Vector(modulo(self.value, other))

    def __iter__(self):
        return iter(self.value)

    def add(self, other, q):
        return (self + other) % q

    def sub(self, other, q):
        # TODO: signed arithmetic needs different logic for underflows
        underflow = self.value < other.value
        result = (self.value - other.value) + underflow.astype(self.value.dtype) * q
        return Vector(result % q)

    def mul(self, other, q):
        return (self * other) % q

    def forward_ntt(self, q, rou):
        """Forward NTT function"""
        q = q.value
        rou2 = (rou * rou) % q
        coefficients = self.value.tolist()
        prefactors = _nb_theory_scratchpad.get_powers_rou(q, len(coefficients), rou=rou)
        for (i, coefficient) in enumerate(coefficients[1:]):
            coefficients[i + 1] = (prefactors[i + 1] * coefficient) % q
        coefficients_ntt = _number_theoretic_transform(coefficients, q, rou=rou2, inverse=False)
        return Vector(np.array(coefficients_ntt, dtype=self.value.dtype))

    def inverse_ntt(self, q, rou):
        """Inverse NTT function"""
        q = q.value
        rou2 = (rou * rou) % q
        coefficients = self.value.tolist()
        coefficients_intt = _number_theoretic_transform(coefficients, q, rou=rou2, inverse=True)
        prefactors = _nb_theory_scratchpad.get_powers_rou(q, len(coefficients), rou)
        for (i, coefficient) in enumerate(coefficients_intt[1:]):
            coefficients_intt[i + 1] = (prefactors[2 * len(coefficients) - i - 1] * coefficient) % q
        return Vector(np.array(coefficients_intt, dtype=self.value.dtype))


@dataclass
class MRP:
    """
    Multi-Residue Polynomial.

    Note: this only supports primes of 32-bits, and unsigned arithmetic.
    """
    # Map each prime to the vector of values. We use 64-bits to support multiplication
    # without extra casts.
    values: dict[np.uint32, npt.NDArray[np.uint64]]

    def __add__(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({
            q: (v1 + other.values[q]) % q for q, v1 in self.values.items()
        })

    def __sub__(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({
            q: (v1 - other.values[q]) % q for q, v1 in self.values.items()
        })

    def __mul__(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({
            q: (v1 * other.values[q]) % q for q, v1 in self.values.items()
        })

    def extract_base(self, base: set[np.uint32]):
        return MRP({q: self.values[q] for q in base})

    def _reconstruct(self, exact: bool) -> Vector:
        big_q = prod(int(q) for q in self.values.keys())
        shape = next(iter(self.values.values())).shape
        result = Vector(np.zeros(shape=shape, dtype=object))
        for q, vec in self.values.items():
            q_star = big_q // int(q)
            q_hat = pow(q_star, -1, int(q))
            vec = intt(vec, q)
            result += Vector(((vec * q_hat) % np.uint64(q)) * q_star)
        if exact:
            result.value %= big_q
        return result

    def extend_base(self, base: set[Scalar], exact: bool):
        common = set(self.values) & base
        if common:
            raise ValueError("Cannot extend to base that is already part of the MRP", common)

        dtype = next(iter(self.values.values())).dtype
        reconstructed = self._reconstruct(exact)
        new_base = {
            q: ntt((reconstructed % q.value).astype(dtype), q)
            for q in base
        }
        return MRP(self.values | new_base)


def ntt(x, q):
    return x

def intt(x, q):
    return x
