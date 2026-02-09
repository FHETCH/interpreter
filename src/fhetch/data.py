"""Data model for FHETCH programs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np


def modulo(x, q):
    x %= q
    if x.dtype == np.int64:
        x -= (x > q//2) * q
    return x


@dataclass(frozen=True)
class Scalar:
    # Use 64-bits exclusively to ensure that there are no 32-bit overflows.
    value: np.int64 | np.uint64

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
        return Vector(modulo(self.value, other).astype(other.dtype))

    def add(self, other, q):
        return (self + other) % q

    def sub(self, other, q):
        return (self - other) % q

    def mul(self, other, q):
        return (self * other) % q


@dataclass
class MRP:
    values: dict[Scalar, Vector]

    def add(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({
            q: v1.add(other.values[q], q) for q, v1 in self.values.items()
        })

    def sub(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({
            q: v1.sub(other.values[q], q) for q, v1 in self.values.items()
        })

    def mul(self, other: MRP):
        assert set(self.values) == set(other.values)
        return MRP({
            q: v1.mul(other.values[q], q) for q, v1 in self.values.items()
        })

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
            raise ValueError("Cannot extend to base that is already part of the MRP", common)

        dtype = next(iter(self.values.values())).value.dtype
        reconstructed = self._reconstruct(exact)
        new_base = {
            q: ntt((reconstructed % q).astype(dtype), q)
            for q in base
        }
        return MRP(self.values | new_base)


def ntt(x, q):
    return x

def intt(x, q):
    return x
