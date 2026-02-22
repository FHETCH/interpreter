"""Data model for FHETCH programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import math

import numpy as np

from .ntt import _nb_theory_scratchpad, _number_theoretic_transform, ROOTS_UNITY


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
        other = getattr(other, 'value', other)
        if isinstance(other, int) and self.value.dtype != np.dtype(object):
            try:
                np.array(other, dtype=self.value.dtype)
            except OverflowError:
                return Vector(self.value.astype(object) * other)
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
    Multi-Residue Polynomial in the evaluation domain.

    Note: this only supports primes of 32-bits, and unsigned arithmetic.
    """
    # Map each prime to the vector of values.
    values: dict[int, Vector]

    @classmethod
    def from_coeffs(cls, base: list[int], coeffs: list[int]):
        degree = len(coeffs)
        coeffs_vec = Vector(np.array(coeffs))
        return cls({
            q: coeffs_vec.forward_ntt(Scalar(q), rou=ROOTS_UNITY.get((degree, q)))
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
        
    def muls(self, other: int|np.integer):
        return MRP({
            q: (v1 * other) % q for q, v1 in self.values.items()
        })

    def extract_base(self, base: set[int]):
        return MRP({q: self.values[q] for q in base})

    def reconstruct(self, exact: bool) -> Vector:
        degree = self.degree()
        for q in self.values.keys():
            if (degree, q) not in ROOTS_UNITY:
                raise RuntimeError("Missing root of unity for (degree, q): ", degree, q)

        big_q = prod(self.values.keys())
        result = Vector(np.zeros(shape=degree, dtype=object))
        for q, vec in self.values.items():
            q_star = big_q // q
            q_hat = Scalar(pow(q_star, -1, q))
            vec = vec.inverse_ntt(Scalar(q), rou=ROOTS_UNITY.get((degree, q)))
            vec = Vector(vec.value.astype(object))
            result += ((vec * q_hat) % q) * q_star

        if exact:
            result.value %= big_q
            # Center the result around 0: values > big_q/2 become negative
            result.value = np.where(result.value > big_q // 2, result.value - big_q, result.value)
        return result

    def extend_base(self, base: set[int], exact: bool):
        common = set(self.values.keys()) & base
        if common:
            raise ValueError("Cannot extend to base that is already part of the MRP", common)

        reconstructed = self.reconstruct(exact)
        degree = len(reconstructed.value)
        for q in base:
            if (degree, q) not in ROOTS_UNITY:
                raise RuntimeError("Missing root of unity for (degree, q): ", degree, q)
        new_base = {
            q: reconstructed.forward_ntt(Scalar(q), rou=ROOTS_UNITY[degree, q])
            for q in base
        }
        return MRP(self.values | new_base)
    
    def degree(self):
        return len(next(iter(self.values.values())).value)
    
    def base(self):
        return set(self.values.keys())
    
    def divq(self, q):
        Q = math.prod(q)
        
        
        not_int_q = self.base() - q
        q_inv = pow(Q, -1, math.prod(not_int_q))
        original = self.extract_base(not_int_q)
        in_q = self.extract_base(q)
        to_sub = in_q.extend_base(not_int_q, True).extract_base(not_int_q)

        result = (original - to_sub).muls(q_inv)

        return result
            
        