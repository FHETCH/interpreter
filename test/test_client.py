import pytest
import numpy as np
from client.context import CryptoContext, Parameters, decode, encode
from client.crypto import Ciphertext
from client.serialization import load_mrp, save_mrp
from client.keygen import gen_sk
from client.utils import find_psi
from fhetch import parser
from fhetch.data import MRP, Vector
from fhetch.env import default_global
from fhetch.eval import eval_func, eval_globals, eval_main
from fhetch.ntt import ROOTS_UNITY


DEFAULT_SCALE = 2.0**29

# Q = [0x10001, 0xC0001]
# P = 0x120001
# PSI = [2, 0xA27C, 0x31779]

# for q, psi in zip(Q + [P], PSI):
#     ROOTS_UNITY[(16, q)] = psi


Q=[0x7ffe0001, 0x7ff80001, 0x7fea0001, 0x7fd20001]
P=[0x7fb40001, 0x7f440001]
for q in Q + P:
    ROOTS_UNITY[(16, q)] = find_psi(q,16)



CUSTOM_PARAMETERS = Parameters(
    # under 32 bits moduli
    q=Q,
    p=P,
    log_n=4,
    log_slots=3,
)


@pytest.fixture
def ctx():
    sk = gen_sk(CUSTOM_PARAMETERS)
    return CryptoContext(params=CUSTOM_PARAMETERS,sk=sk)


def test_encode_decode(ctx):
    """Test that encoding and decoding a message returns the original message."""
    # Create a test message
    msg = np.array([1, 2, 5, 2, -1, 0, 4.5, 1.5])

    # Encrypt the message
    pt = encode(msg, DEFAULT_SCALE, ctx._params.moduli)

    # Decrypt the ciphertext
    dec_msg = decode(pt)

    # Check that the decrypted message matches the original
    # Use allclose for floating point comparison with tolerance
    np.testing.assert_allclose(dec_msg, msg, rtol=1e-3, atol=1e-3)


def test_encrypt_decrypt(ctx: CryptoContext):
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


def test_add(ctx:CryptoContext):
    import os, shutil
    os.makedirs("temp", exist_ok=True)
    try:
        msg = np.arange(1, 9)
        
        ciphertext = ctx.encrypt_msg(list(msg), DEFAULT_SCALE)
        
        save_mrp(ciphertext.polynomials[0],"temp/ct_0.npz")
        save_mrp(ciphertext.polynomials[1],"temp/ct_1.npz")

        prog = parser.Program.parse_string(
            """
        primes Q = [0x10001, 0xC0001];
        def main() {
            var ct_0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("temp/ct_0.npz");
            var ct_1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("temp/ct_1.npz");
            var s0 = ct_0 + ct_0(mod Q);
            var s1 = ct_1 + ct_1(mod Q);
            write_mrp_u32_1024_Q(s0,"temp/ct_res_0.npz");
            write_mrp_u32_1024_Q(s1,"temp/ct_res_1.npz");
        }
        """
        ).program
        global_env = default_global()
        global_env.update(eval_globals(prog))
        eval_main(prog, global_env)
        ct_res_0 = load_mrp("temp/ct_res_0.npz")
        ct_res_1 = load_mrp("temp/ct_res_1.npz")
        decrypted_msg = ctx.decrypt_msg(Ciphertext(DEFAULT_SCALE,[ct_res_0,ct_res_1]))
        np.testing.assert_allclose(decrypted_msg, msg+msg, rtol=1e-3, atol=1e-3)
    finally:
        shutil.rmtree("temp")
        
        
def test_mult(ctx:CryptoContext):
    import os, shutil
    os.makedirs("temp", exist_ok=True)
    try:
        msg = np.arange(1, 9)
        msg_3 = np.array([3 for _ in range(8)])
        
        ciphertext = ctx.encrypt_msg(list(msg), DEFAULT_SCALE)
        ciphertext_3 = ctx.encrypt_msg(list(msg_3), DEFAULT_SCALE)
        
        save_mrp(ciphertext.polynomials[0],"temp/ct_0.npz")
        save_mrp(ciphertext.polynomials[1],"temp/ct_1.npz")
        
        save_mrp(ciphertext.polynomials[0],"temp/ct3_0.npz")
        save_mrp(ciphertext.polynomials[1],"temp/ct3_1.npz")

        prog = parser.Program.parse_string(
            """
        primes Digit0 = [0x7ffe0001, 0x7ff80001];
        primes Digit1 = [0x7fea0001, 0x7fd20001];
        primes Q = Digit0||Digit1;
        primes P = [0x7fb40001, 0x7f440001];
        primes QP = Q||P;


        def KeySwitch(poly: MRP<u32, 1024, Q>, k00, k01, k10, k11) { 
            var decomposed0 = BaseExtend(poly, Digit0, QP);
            var decomposed1 = BaseExtend(poly, Digit1, QP);

            var accumulator0 = decomposed0 * k00 + decomposed1 * k10;
            var accumulator1 = decomposed0 * k01 + decomposed1 * k11;

            return [Rescale(accumulator0, P), Rescale(accumulator1, P)];
        }

        def main() {
            var ct_a0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("ct_a0.npz");
            var ct_a1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("ct_a1.npz");

            var ct_b0: MRP<u32, 1024, Q> = read("ct_b0.npz");
            var ct_b1: MRP<u32, 1024, Q> = read("ct_b1.npz");

            var prod0 = ct_a0 * ct_b0;
            var prod1 = ct_a0 * ct_b1 + ct_a1 * ct_b0;
            var prod2 = ct_a1 * ct_b1;

            var relin_d0_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_QP();
            var relin_d0_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_QP();
            var relin_d1_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_QP();
            var relin_d1_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_QP();

            var ks = KeySwitch(prod2, relin_d0_0, relin_d0_1, relin_d1_0, relin_d1_1);
            write(prod0 + get(ks, 0));
            write(prod1 + get(ks, 1));
        }
        """
        ).program
        global_env = default_global()
        global_env.update(eval_globals(prog))
        eval_main(prog, global_env)
        ct_res_0 = load_mrp("temp/ct_res_0.npz")
        ct_res_1 = load_mrp("temp/ct_res_1.npz")
        decrypted_msg = ctx.decrypt_msg(Ciphertext(DEFAULT_SCALE,[ct_res_0,ct_res_1]))
        np.testing.assert_allclose(decrypted_msg, msg+msg, rtol=1e-3, atol=1e-3)
    finally:
        shutil.rmtree("temp")
