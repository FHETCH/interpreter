import pytest
import numpy as np
from client.crypto_context import CryptoContext, Parameters, decode, encode
from client.encode import embedding, unpacking
from fhetch.ntt import ROOTS_UNITY


DEFAULT_SCALE = 2.0**29

Q = [0x10001, 0xc0001]
P = 0x120001
PSI = [2, 0xa27c, 0x31779]

for q, psi in zip(Q + [P], PSI):
    ROOTS_UNITY[(16, q)] = psi

CUSTOM_PARAMETERS = Parameters(
    # under 32 bits moduli
    q=Q,
    p=[P],
    log_n=4,
    log_slots=3,
)

@pytest.fixture
def ctx():
    return CryptoContext(params=CUSTOM_PARAMETERS)



def test_encode_decode(ctx):
    """Test that encrypting and decrypting a message returns the original message."""
    # Create a test message 
    msg = np.array([1, 2, 5, 2, -1, 0, 4.5, 1.5])

    # Encrypt the message
    pt = encode(msg, DEFAULT_SCALE,ctx._params.moduli )

    # Decrypt the ciphertext
    dec_msg = decode(pt)

    # Check that the decrypted message matches the original
    # Use allclose for floating point comparison with tolerance
    np.testing.assert_allclose(dec_msg, msg, rtol=1e-3, atol=1e-3)


def test_encrypt_decrypt(ctx):
    """Test that encrypting and decrypting a message returns the original message."""
    # Create a test message 
    msg = np.array([1, 2, 5, 2, -1, 0, 4.5, 1.5])

    # Encrypt the message
    ciphertext = ctx.encrypt_msg(msg, DEFAULT_SCALE)

    # Decrypt the ciphertext
    decrypted_msg = ctx.decrypt_msg(ciphertext)

    # Check that the decrypted message matches the original
    # Use allclose for floating point comparison with tolerance
    np.testing.assert_allclose(decrypted_msg, msg, rtol=1e-3, atol=1e-3)
