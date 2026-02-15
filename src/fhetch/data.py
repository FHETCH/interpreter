"""Data model for FHETCH programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from .ntt import _nb_theory_scratchpad, _number_theoretic_transform


def modulo(x, q):
    x %= q
    if x.dtype == np.int64:
        x -= (x > q // 2) * q
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

    # TODO: Add negate


@dataclass
class Vector:
    value: np.array

    def __add__(self, other):
        return Vector(self.value + other.value)

    def __sub__(self, other):
        return Vector(self.value - other.value)

    def __mul__(self, other: Vector | Scalar):
        return self.inner_mul(other, q=None)
    
    def inner_mul(self, other: Vector | Scalar, q:int):
        if isinstance(other, Scalar):
            result = Vector(self.value * other.value)
            if q is not None:
                result = Vector(modulo(result.value ,q))
        elif self.is_flat() and other.is_flat():
            result = Vector(self.value * other.value)
            if q is not None:
                result = Vector(modulo(result.value ,q))
        elif not self.is_flat() and other.is_flat():
            # Broadcasting: vector of vectors * flat vector
            # Each sub-vector gets multiplied by corresponding scalar
            result = Vector(np.array([a.inner_mul(Scalar(int(b)), q) for a, b in zip(self.value, other.value)], dtype=object))
        else:
            result = Vector(np.array([a.inner_mul(b, q) for a, b in zip(self.value, other.value)], dtype=object))
      
        return result

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

    def mul(self, other, q):
        if isinstance(q,Scalar):
            q=q.value
        return self.inner_mul(other, q)

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
    def is_flat(self):
        return isinstance(self.value[0],np.integer)

@dataclass
class MRP:
    values: dict[Scalar, Vector]

    def add(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({q: v1.add(other.values[q], q) for q, v1 in self.values.items()})

    def sub(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({q: v1.sub(other.values[q], q) for q, v1 in self.values.items()})

    def mul(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({q: v1.mul(other.values[q], q) for q, v1 in self.values.items()})

    def extract_base(self, base: set[Scalar]):
        return MRP({q: self.values[q] for q in base})

    def _reconstruct(self, exact: bool):
        big_q = prod(q.value for q in self.values.keys())
        shape = next(iter(self.values.values())).value.shape
        result = Vector(np.zeros(shape=shape, dtype=object))
        for q, vec in self.values.items():
            q_star = big_q // q.value
            q_hat = Scalar(pow(q_star, -1, q.value))
            print(f"{q_star=}")
            print(f"{q_hat=}")
            vec = intt(vec, q)
            result += vec.mul(q_hat, Scalar(q)) * Scalar(q_star)
            print(f"{result=}")
        if exact:
            result.value %= big_q
        return result

    def extend_base(self, base: set[Scalar], exact: bool):
        common = set(self.values) & base
        if common:
            raise ValueError(
                "Cannot extend to base that is already part of the MRP", common
            )

        dtype = next(iter(self.values.values())).value.dtype
        reconstructed = self._reconstruct(exact)
        new_base = {q: ntt((reconstructed % q).astype(dtype), q) for q in base}
        return MRP(self.values | new_base)


def ntt(x, q):
    return x


def intt(x, q):
    return x
