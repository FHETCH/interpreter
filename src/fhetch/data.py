"""Data model for FHETCH programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from .ntt import _nb_theory_scratchpad, _number_theoretic_transform, ROOTS_UNITY


def modulo(x, q):
    x %= q
    if x.dtype == np.int64:
        x -= (x > q // 2) * q
    return x

def mul_reorder(lhs: Vector, rhs: Vector):
    if isinstance(lhs.value[0], np.integer):
        return rhs, lhs
    return lhs, rhs

@dataclass(frozen=True)
class Scalar:
    # We use Python's arbitrary precision int for scalars in order to allow
    # arbitrary computations to produce primes.
    value: int

    def __add__(self, other):
        if isinstance(other, Vector):
            raise TypeError("Cannot add Vector to Scalar")
        return Scalar(self.value + other.value)

    def __sub__(self, other):
        if isinstance(other, Vector):
            raise TypeError("Cannot Subtract Vector from Scalar")
        return Scalar(self.value - other.value)

    def __mul__(self, other):
        if isinstance(other, Vector):
            raise TypeError("Cannot Multiply Scalar with Vector")
        return Scalar(self.value * other.value)

    def __mod__(self, q):
        return Scalar(self.value % q.value)

    def add(self, other, q):
        if isinstance(other, Vector):
            return Vector(modulo(other.value + self.value, q.value))
        return Scalar(modulo(self.value + other.value, q.value))

    def sub(self, other, q):
        if isinstance(other, Vector):
            result = self.value - other.value
            return Vector(modulo(result, q.value))
        return Scalar(modulo(self.value - other.value, q.value))

    def mul(self, other, q):
        if isinstance(other, Vector):
            return other.mul(self, q)
        return Scalar(modulo(self.value * other.value, q.value))

    # TODO: Add negate

@dataclass
class Vector:
    value: np.array

    def __add__(self, other: Vector | Scalar | int):
        if isinstance(other, int):
            return Vector(self.value + other)
        # if Vector or Scalar
        return Vector(self.value + other.value)

    def __sub__(self, other: Vector | Scalar | int):
        if isinstance(other, int):
            return Vector(self.value - other)
        # if Vector or Scalar
        return Vector(self.value - other.value)

    def __mul__(self, other):
        other = getattr(other, 'value', other)
        return Vector(self.value * other)

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

    # TODO: Add negate

    def mul(self, other, q=None):
        if isinstance(q, Scalar):
            q = q.value
        # Convert scalar to python int
        if isinstance(other, Scalar):
            other = other.value
        # Vector * Scalar
        if isinstance(other, np.integer | int):
            result = Vector(self.value * other)
        else:
            # Vector * Vector
            lhs, rhs = mul_reorder(self, other)
            result = Vector(lhs.value * rhs.value)
        if q is not None:
            result = result % q
        return result

    def forward_ntt(self, q, rou):
        """Forward NTT function"""
        q = q.value
        rou2 = (rou * rou) % q
        coefficients = self.value.tolist()
        prefactors = _nb_theory_scratchpad.get_powers_rou(q, len(coefficients), rou=rou)
        for i, coefficient in enumerate(coefficients[1:]):
            coefficients[i + 1] = (prefactors[i + 1] * coefficient) % q
        coefficients_ntt = _number_theoretic_transform(
            coefficients, q, rou=rou2, inverse=False
        )
        return Vector(np.array(coefficients_ntt, dtype=self.value.dtype))

    def inverse_ntt(self, q, rou):
        """Inverse NTT function"""
        q = q.value
        rou2 = (rou * rou) % q
        coefficients = self.value.tolist()
        coefficients_intt = _number_theoretic_transform(
            coefficients, q, rou=rou2, inverse=True
        )
        prefactors = _nb_theory_scratchpad.get_powers_rou(q, len(coefficients), rou)
        for i, coefficient in enumerate(coefficients_intt[1:]):
            coefficients_intt[i + 1] = (
                prefactors[2 * len(coefficients) - i - 1] * coefficient
            ) % q
        return Vector(np.array(coefficients_intt, dtype=self.value.dtype))


@dataclass
class MRP:
    """
    Multi-Residue Polynomial in the evaluation domain.

    Note: this only supports primes of 32-bits, and unsigned arithmetic.
    """
    # Map each prime to the vector of values.
    values: dict[int, Vector]

    @classmethod
    def from_coeffs(cls, base: list[int], coeffs: list[int]):
        degree = len(coeffs)
        coeffs = Vector(np.array(coeffs))
        return cls({
            q: coeffs.forward_ntt(Scalar(q), rou=ROOTS_UNITY.get((degree, q)))
            for q in base
        })

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

    def extract_base(self, base: set[int]):
        return MRP({q: self.values[q] for q in base})

    def reconstruct(self, exact: bool) -> Vector:
        degree = len(next(iter(self.values.values())).value)
        for q in self.values.keys():
            if (degree, q) not in ROOTS_UNITY:
                raise RuntimeError("Missing root of unity for (degree, q): ", degree, q)

        big_q = prod(self.values.keys())
        result = Vector(np.zeros(shape=degree, dtype=object))
        for q, vec in self.values.items():
            q_star = big_q // q
            q_hat = Scalar(pow(q_star, -1, q))
            vec = vec.inverse_ntt(Scalar(q), rou=ROOTS_UNITY.get((degree, q)))
            result += ((vec * q_hat) % q) * q_star

        if exact:
            result.value %= big_q
        return result

    def extend_base(self, new_primes: set[int], exact: bool):
        common = set(self.values.keys()) & new_primes
        if common:
            raise ValueError(
                "Cannot extend to base that is already part of the MRP", common
            )

        reconstructed = self.reconstruct(exact)
        degree = len(reconstructed.value)
        for q in new_primes:
            if (degree, q) not in ROOTS_UNITY:
                raise RuntimeError("Missing root of unity for (degree, q): ", degree, q)
        new_base = {
            q: reconstructed.forward_ntt(Scalar(q), rou=ROOTS_UNITY[degree, q])
            for q in new_primes
        }
        return MRP(self.values | new_base)
