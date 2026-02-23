from random import randint
from math import prod

import numpy as np

from fhetch.data import MRP
from fhetch.ntt import ROOTS_UNITY


Q = [0x10001, 0xc0001]
P = 0x120001
PSI = [2, 0xa27c, 0x31779]

for q, psi in zip(Q + [P], PSI):
    ROOTS_UNITY[(16, q)] = psi


def test_from_coeffs():
    poly = MRP.from_coeffs(Q, list(range(16)))
    reconstructed = poly.reconstruct(exact=True)
    assert np.all(reconstructed.value == list(range(16)))


def test_signed_coeffs():
    # Useful for secret keys that have signed coefficients
    coeffs = [0, 1, -1, 0] * 4
    poly = MRP.from_coeffs(Q, coeffs)
    reconstructed = poly.reconstruct(exact=True)
    assert np.all(reconstructed.value == coeffs)


def test_base_extend():
    coeffs = [randint(0, 100) for _ in range(16)]
    poly = MRP.from_coeffs(Q, coeffs)
    new = poly.extend_base({P}, exact=True)
    assert np.all(new.reconstruct(True).value == coeffs)

    # fast base extensions add a factor to the output
    new = poly.extend_base({P}, exact=False)
    expected_diff = prod(Q)
    new_coeffs = new.reconstruct(False)
    assert np.all((new_coeffs.value - coeffs) % expected_diff == 0)
