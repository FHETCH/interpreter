import numpy as np
from sympy import primitive_root

from fhetch.data import MRP, Vector


def random_poly(base: list[int], degree: int) -> MRP:
    rng = np.random.default_rng()
    return MRP({
        q: Vector(rng.integers(0, q, size=degree))
        for q in base
    })


def find_psi(q: int, n: int) -> int:
    """Find a primitive 2n-th root of unity mod q for negacyclic NTT.

    In CKKS, the negacyclic NTT over Z_q[x]/(x^n + 1) requires a
    primitive 2n-th root of unity psi such that psi^(2n) ≡ 1 (mod q)
    and psi^k ≢ 1 for any 0 < k < 2n.

    Requirements:
        - q must be prime
        - q ≡ 1 (mod 2n)  (so that a 2n-th root of unity exists in Z_q*)

    Algorithm:
        g  = primitive root of Z_q*  (generator of the full cyclic group of order q-1)
        psi = g ^ ((q-1) / (2n))  mod q

    Args:
        q: NTT-friendly prime modulus.
        n: Ring dimension (polynomial degree).

    Returns:
        A primitive 2n-th root of unity mod q.
    """
    two_n = 2 * n
    if (q - 1) % two_n != 0:
        raise ValueError(
            f"q={q} does not support a primitive {two_n}-th root of unity: "
            f"(q-1) % 2n = {(q - 1) % two_n} (need 0)"
        )
    g = primitive_root(q)
    psi = pow(g, (q - 1) // two_n, q)
    return psi