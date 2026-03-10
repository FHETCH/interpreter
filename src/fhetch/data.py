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
        x -= (x > q // 2) * q
    return x

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

    def __mod__(self, q:Scalar|int|np.integer):
        if isinstance(q, Scalar):
            q = q.value
        return Scalar(self.value % q)

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
        if isinstance(other, int) and self.value.dtype != np.dtype(object) and other >= 1<<32 :
           raise OverflowError("Potential Overflow")
                
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

    def mul(self, other, q):
        other = other % q
        return (self * other) % q
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def forward_ntt(self, q, rou):
        """Forward NTT function"""
        q = q.value
        rou2 = (rou * rou) % q
        n = len(self.value)
        psi_powers = _nb_theory_scratchpad.get_powers_rou(q, n, rou=rou)
        # Reduce mod q first: coefficients may be unreduced (e.g. from special_ifft)
        # or stored as object-dtype big ints. Without this, int64 overflows.
        # After reduction both operands are < q < 2^31, so their product < 2^62 < INT64_MAX.
        twisted = (self.value % q).astype(np.int64) * psi_powers[:n] % q
        coefficients_ntt = _number_theoretic_transform(twisted, q, rou=rou2, inverse=False)
        return Vector(coefficients_ntt.astype(self.value.dtype))

    def inverse_ntt(self, q, rou):
        """Inverse NTT function"""
        q = q.value
        rou2 = (rou * rou) % q
        n = len(self.value)
        coefficients_intt = _number_theoretic_transform(self.value, q, rou=rou2, inverse=True)
        psi_powers = _nb_theory_scratchpad.get_powers_rou(q, n, rou)
        # Negacyclic post-twist: multiply result[i] by psi^(2n - i)  (i = 1..n-1, index 2n-1 down to n+1)
        intt_arr = coefficients_intt.astype(np.int64)
        # post-twist indices: 0 stays as-is; for i>=1 use powers[2n - i]
        twist_idx = np.arange(n, dtype=np.int64)
        twist_idx[1:] = 2 * n - twist_idx[1:]
        intt_arr = intt_arr * psi_powers[twist_idx] % q
        return Vector(intt_arr.astype(self.value.dtype))
    
    def ntt_automorphism(self,g):
        """
        Permutes coefficients already in the NTT domain.
        ntt_coeffs: 1D array of size N
        g: Galois element (e.g., 5^rot)
        """
        N = len(self.value)
        new_ntt = np.zeros(N, dtype=self.value.dtype)
        
        for i in range(N):
            # Map index to the odd power of the root of unity
            old_exponent = 2 * i + 1
            # Apply the automorphism mapping
            new_exponent = (old_exponent * g) % (2 * N)
            # Map back to the array index
            new_index = (new_exponent - 1) // 2
            
            new_ntt[new_index] = self.value[i]
            
        return Vector(new_ntt) 


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
        
    def muls(self, other: int|np.integer):
        return MRP({
            q: (v1 * other) % q for q, v1 in self.values.items()
        })

    def extract_base(self, base: set[int]):
        return MRP({q: self.values[q] for q in base})

    def reconstruct(self, exact: bool,signed:bool) -> Vector:
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
            vec.value = vec.value.astype(object)
            result += vec.mul(q_hat,q) * q_star

        if exact:
            result.value %= big_q
            if signed:
                result.value = np.where(result.value > big_q // 2, result.value - big_q, result.value)

        return result

    def extend_base(self, new_primes: set[int], exact: bool):
        common = set(self.values.keys()) & new_primes
        if common:
            raise ValueError(
                "Cannot extend to base that is already part of the MRP", common
            )

        reconstructed = self.reconstruct(exact,signed=False)
        degree = len(reconstructed.value)
        for q in new_primes:
            if (degree, q) not in ROOTS_UNITY:
                raise RuntimeError("Missing root of unity for (degree, q): ", degree, q)
        new_base = {
            q: reconstructed.forward_ntt(Scalar(q), rou=ROOTS_UNITY[degree, q])
            for q in new_primes
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
        to_sub = in_q.extend_base(not_int_q, exact=False).extract_base(not_int_q)

        result = (original - to_sub).muls(q_inv)
        return result
    def automorph(self, rot:int):
        degree = self.degree()
        return MRP({q: self.values[q].ntt_automorphism(pow(5,rot,2*degree)) for q in self.base()})
    

            
        