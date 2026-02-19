from sympy.ntheory import isprime
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int


# Scratchpad to store powers of roots of unity
#
# NOTE: This implementation assumes the root of unity is set the first time an (i)NTT is called
# for a given size and modulus. The same root is re-used for all subsequent (i)NTTs with the same
# dimension and modulus.
class _NbTheoryScratchpad:

    def __init__(self):
        self.powers_rou = {}

    def add_powers_rou(self, modulus: int, ring_dimension: int, rou: int):
        w = 1
        powers = [w]
        for i in range(1, 2 * ring_dimension):
            w = (w * rou) % modulus
            powers.append(w)
        self.powers_rou[(modulus, ring_dimension)] = powers

    def get_powers_rou(self, modulus: int, ring_dimension: int, rou: int):
        if not (modulus, ring_dimension) in self.powers_rou:
            self.add_powers_rou(modulus, ring_dimension, rou=rou)
        return self.powers_rou[(modulus, ring_dimension)]


_nb_theory_scratchpad = _NbTheoryScratchpad()

# Root of unity used for the NTT (if None, use the default from Sympy)
ROOTS_UNITY = {}


# Modified NTT function from Sympy
def _number_theoretic_transform(seq, prime, rou, inverse=False):
    """Utility function for the Number Theoretic Transform"""

    if not iterable(seq):
        raise TypeError("Expected a sequence of integer coefficients "
                        "for Number Theoretic Transform")

    p = as_int(prime)
    if not isprime(p):
        raise ValueError("Expected prime modulus for "
                        "Number Theoretic Transform")

    a = [as_int(x) % p for x in seq]

    n = len(a)
    if n < 1:
        return a

    b = n.bit_length() - 1
    if n&(n - 1):
        b += 1
        n = 2**b

    if (p - 1) % n:
        raise ValueError("Expected prime modulus of the form (m*2**k + 1)")

    a += [0]*(n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            a[i], a[j] = a[j], a[i]

    # The root of unity to be effectively used should be the inverse if 
    # computing an iNTT
    rt = rou if not inverse else pow(rou, -1, p)

    w = [1]*(n // 2)
    for i in range(1, n // 2):
        w[i] = w[i - 1]*rt % p

    h = 2
    while h <= n:
        hf, ut = h // 2, n // h
        for i in range(0, n, h):
            for j in range(hf):
                u, v = a[i + j], a[i + j + hf]*w[ut * j]
                a[i + j], a[i + j + hf] = (u + v) % p, (u - v) % p
        h *= 2

    if inverse:
        rv = pow(n, p - 2, p)
        a = [x*rv % p for x in a]

    return a
