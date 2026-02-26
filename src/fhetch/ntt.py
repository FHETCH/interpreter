from sympy.ntheory import isprime

import numpy as np



# Scratchpad to store powers of roots of unity
#
# NOTE: This implementation assumes the root of unity is set the first time an (i)NTT is called
# for a given size and modulus. The same root is re-used for all subsequent (i)NTTs with the same
# dimension and modulus.
def _build_bit_reversal(n: int, b: int) -> np.ndarray:
    """Return the bit-reversal permutation index array for length n = 2**b."""
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    tmp = idx.copy()
    for _ in range(b):
        rev = (rev << 1) | (tmp & 1)
        tmp >>= 1
    return rev


class _NbTheoryScratchpad:

    def __init__(self):
        self.powers_rou = {}   # (modulus, ring_dimension) -> np.ndarray of int64
        self._bit_rev = {}     # n -> np.ndarray index array

    def add_powers_rou(self, modulus: int, ring_dimension: int, rou: int):
        # Build the full power table as a numpy int64 array using Horner-style
        # multiplication so we stay in Python bigints only for the seed.
        length = 2 * ring_dimension
        powers = np.empty(length, dtype=np.int64)
        w = 1
        for i in range(length):
            powers[i] = w
            w = (w * rou) % modulus
        self.powers_rou[(modulus, ring_dimension)] = powers

    def get_powers_rou(self, modulus: int, ring_dimension: int, rou: int) -> np.ndarray:
        if (modulus, ring_dimension) not in self.powers_rou:
            self.add_powers_rou(modulus, ring_dimension, rou=rou)
        return self.powers_rou[(modulus, ring_dimension)]

    def get_bit_rev(self, n: int, b: int) -> np.ndarray:
        if n not in self._bit_rev:
            self._bit_rev[n] = _build_bit_reversal(n, b)
        return self._bit_rev[n]


_nb_theory_scratchpad = _NbTheoryScratchpad()

# Root of unity used for the NTT (if None, use the default from Sympy)
ROOTS_UNITY = {}


def _number_theoretic_transform(seq, prime, rou, inverse=False):
    """Vectorized Number Theoretic Transform using NumPy.

    Primes must be ≤ 2**31 so that products fit in int64 (p² < 2**63).
    """
    p = int(prime)
    if not isprime(p):
        raise ValueError("Expected prime modulus for Number Theoretic Transform")

    # --- input as int64 array ------------------------------------------------
    a = np.asarray(seq, dtype=np.int64) % p

    n = len(a)
    if n < 1:
        return a

    b = n.bit_length() - 1
    if n & (n - 1):        # not a power of two — pad
        b += 1
        n = 1 << b
        a = np.resize(a, n)
        a[len(seq):] = 0

    if (p - 1) % n:
        raise ValueError("Expected prime modulus of the form (m*2**k + 1)")

    # --- bit-reversal permutation (cached) -----------------------------------
    rev = _nb_theory_scratchpad.get_bit_rev(n, b)
    a = a[rev]

    # --- twiddle factor table -------------------------------------------------
    # rt = rou for forward NTT, rou⁻¹ for inverse
    rt = rou if not inverse else pow(int(rou), -1, p)
    # Build the n//2 twiddle factors w[i] = rt^i mod p
    w = np.empty(n // 2, dtype=np.int64)
    w[0] = 1
    for i in range(1, n // 2):
        w[i] = w[i - 1] * rt % p

    # --- butterfly stages (log2(n) Python iterations, all work vectorized) ---
    h = 2
    while h <= n:
        hf = h >> 1
        ut = n // h
        # twiddle factors for this stage: w[0], w[ut], w[2*ut], ..., w[(hf-1)*ut]
        tw = w[np.arange(hf, dtype=np.int64) * ut]   # shape (hf,)
        # reshape into (n//h, h) blocks so each row is one butterfly group
        a = a.reshape(-1, h)                          # view
        u = a[:, :hf].copy()
        v = a[:, hf:] * tw % p                        # broadcast over blocks
        a[:, :hf] = (u + v) % p
        a[:, hf:] = (u - v) % p
        a = a.reshape(n)
        h <<= 1

    # --- iNTT final scaling --------------------------------------------------
    if inverse:
        rv = pow(n, p - 2, p)
        a = a * rv % p

    return a
