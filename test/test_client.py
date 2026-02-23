import pytest
import numpy as np
from client.context import CryptoContext, Parameters, decode, encode
from client.keygen import gen_sk
from client.utils import find_psi
from fhetch.ntt import ROOTS_UNITY

Q = [
    0x7FFFD801,
    0x7FFE9001,
    0x7FFE8801,
    0x7FFE6001,
    0x7FFE1801,
    0x7FFE0001,
    0x7FFDE801,
    0x7FFDC801,
    0x7FFD5801,
    0x7FFD2801,
    0x7FFD2001,
    0x7FFC4801,
    0x7FFC3801,
    0x7FFBF001,
    0x7FFBC001,
    0x7FFBA001,
    0x7FFB5801,
    0x7FFA2801,
    0x7FF9E001,
    0x7FF9C001,
    0x7FF96801,
    0x7FF94801,
    0x7FF93801,
    0x7FF87001,
    0x7FF81001,
]

P = [0x7FF80001, 0x7FF7B001, 0x7FF73801, 0x7FF6E001, 0x7FF6A801]


FHETCH_PRIMES = """
    primes Digit0 = [0x7FFFD801, 0x7FFE9001, 0x7FFE8801, 0x7FFE6001, 0x7FFE1801];
    primes Digit1 = [0x7FFE0001, 0x7FFDE801, 0x7FFDC801, 0x7FFD5801, 0x7FFD2801];
    primes Digit2 = [0x7FFD2001, 0x7FFC4801, 0x7FFC3801, 0x7FFBF001, 0x7FFBC001];
    primes Digit3 = [0x7FFBA001, 0x7FFB5801, 0x7FFA2801, 0x7FF9E001, 0x7FF9C001];
    primes Digit4 = [0x7FF96801, 0x7FF94801, 0x7FF93801, 0x7FF87001, 0x7FF81001];
    primes Q = Digit0||Digit1||Digit2||Digit3||Digit4;
    primes P = [0x7FF80001, 0x7FF7B001, 0x7FF73801, 0x7FF6E001, 0x7FF6A801];
    primes QP = Q||P;
"""

for q in Q + P:
    ROOTS_UNITY[(1024, q)] = find_psi(q, 1024)

CUSTOM_PARAMETERS = Parameters(
    # under 32 bits moduli
    q=Q,
    p=P,
    log_n=10,
    log_slots=9,
)


@pytest.fixture
def ctx():
    sk = gen_sk(CUSTOM_PARAMETERS)
    return CryptoContext(params=CUSTOM_PARAMETERS, sk=sk)


def test_encode_decode(ctx):
    """Test that encoding and decoding a message returns the original message."""
    # Create a test message
    msg = np.random.randint(0, np.iinfo(np.int16).max, size=ctx._params.slots)

    # Encrypt the message
    pt = encode(msg, ctx._params.scaling_factor(), ctx._params.moduli)

    # Decrypt the ciphertext
    dec_msg = decode(pt,ctx._params.scaling_factor())

    # Check that the decrypted message matches the original
    # Use allclose for floating point comparison with tolerance
    np.testing.assert_allclose(dec_msg, msg, rtol=1e-3, atol=1e-3)

