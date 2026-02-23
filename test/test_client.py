import pytest
import numpy as np
from client.context import CryptoContext, Parameters, decode, encode
from client.crypto import Ciphertext
from client.serialization import load_mrp, save_mrp
from client.keygen import gen_ksk, gen_relin_key, gen_sk
from client.utils import find_psi
from fhetch import parser
from fhetch.data import MRP, Vector
from fhetch.env import default_global
from fhetch.eval import eval_func, eval_globals, eval_main
from fhetch.ntt import ROOTS_UNITY


DEFAULT_SCALE = 2.0**31

# Q = [0x10001, 0xC0001]
# P = 0x120001
# PSI = [2, 0xA27C, 0x31779]

# for q, psi in zip(Q + [P], PSI):
#     ROOTS_UNITY[(16, q)] = psi


Q = [
    0x7FFFFF61, 0x7FFFFE01, 0x7FFFFCC1, 0x7FFFFAA1, 0x7FFFF9E1,
    0x7FFFF8C1, 0x7FFFF541, 0x7FFFF441, 0x7FFFF261, 0x7FFFF181,
    0x7FFFF081, 0x7FFFEFC1, 0x7FFFEF41, 0x7FFFECC1, 0x7FFFEBE1,
    0x7FFFEA21, 0x7FFFEA01, 0x7FFFE9C1, 0x7FFFE7E1, 0x7FFFE701,
    0x7FFFE5A1, 0x7FFFE521, 0x7FFFE3C1, 0x7FFFE361, 0x7FFFE101,
]
P = [0x7FFFE061, 0x7FFFE041, 0x7FFFDF21, 0x7FFFDDC1, 0x7FFFDCE1]
for q in Q + P:
    ROOTS_UNITY[(16, q)] = find_psi(q, 16)


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
    return CryptoContext(params=CUSTOM_PARAMETERS, sk=sk)


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


def test_add(ctx: CryptoContext):
    import os, shutil

    os.makedirs("temp", exist_ok=True)
    try:
        msg = np.arange(1, 9)

        ciphertext = ctx.encrypt_msg(list(msg), DEFAULT_SCALE)

        save_mrp(ciphertext.polynomials[0], "temp/ct_0.npz")
        save_mrp(ciphertext.polynomials[1], "temp/ct_1.npz")

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
        decrypted_msg = ctx.decrypt_msg(Ciphertext(DEFAULT_SCALE, [ct_res_0, ct_res_1]))
        np.testing.assert_allclose(decrypted_msg, msg + msg, rtol=1e-3, atol=1e-3)
    finally:
        shutil.rmtree("temp")


def test_mult(ctx: CryptoContext):
    import os, shutil

    try:
        os.makedirs("temp", exist_ok=True)
        msg1 = np.random.randint(0, np.iinfo(np.int16).max, size=8)
        msg2 = np.random.randint(0, np.iinfo(np.int16).max, size=8)

        ciphertext = ctx.encrypt_msg(list(msg1), DEFAULT_SCALE)
        ciphertext_3 = ctx.encrypt_msg(list(msg2), DEFAULT_SCALE)

        save_mrp(ciphertext.polynomials[0], "temp/ct_a0.npz")
        save_mrp(ciphertext.polynomials[1], "temp/ct_a1.npz")

        save_mrp(ciphertext_3.polynomials[0], "temp/ct_b0.npz")
        save_mrp(ciphertext_3.polynomials[1], "temp/ct_b1.npz")

        relin_key = gen_relin_key(ctx._sk, Q, P)

        for i, (ksk_0, ksk_1) in enumerate(relin_key):
            save_mrp(ksk_0, f"temp/relin_d{i}_0.npz")
            save_mrp(ksk_1, f"temp/relin_d{i}_1.npz")

        prog = parser.Program.parse_string(
            """
        primes Digit0 = [0x7FFFFF61, 0x7FFFFE01, 0x7FFFFCC1, 0x7FFFFAA1, 0x7FFFF9E1];
        primes Digit1 = [0x7FFFF8C1, 0x7FFFF541, 0x7FFFF441, 0x7FFFF261, 0x7FFFF181];
        primes Digit2 = [0x7FFFF081, 0x7FFFEFC1, 0x7FFFEF41, 0x7FFFECC1, 0x7FFFEBE1];
        primes Digit3 = [0x7FFFEA21, 0x7FFFEA01, 0x7FFFE9C1, 0x7FFFE7E1, 0x7FFFE701];
        primes Digit4 = [0x7FFFE5A1, 0x7FFFE521, 0x7FFFE3C1, 0x7FFFE361, 0x7FFFE101];
        primes Q = Digit0||Digit1||Digit2||Digit3||Digit4;
        primes P = [0x7FFFE061, 0x7FFFE041, 0x7FFFDF21, 0x7FFFDDC1, 0x7FFFDCE1];
        primes QP = Q||P;


        def KeySwitch(poly: MRP<u32, 1024, Q>, k00, k01, k10, k11, k20, k21, k30, k31, k40, k41) { 
            var decomposed0 = BaseExtend(poly, Digit0, QP);
            var decomposed1 = BaseExtend(poly, Digit1, QP);
            var decomposed2 = BaseExtend(poly, Digit2, QP);
            var decomposed3 = BaseExtend(poly, Digit3, QP);
            var decomposed4 = BaseExtend(poly, Digit4, QP);

            var accumulator0 = decomposed0 * k00 + decomposed1 * k10 + decomposed2 * k20 + decomposed3 * k30 + decomposed4 * k40;
            var accumulator1 = decomposed0 * k01 + decomposed1 * k11 + decomposed2 * k21 + decomposed3 * k31 + decomposed4 * k41;

            return [Rescale(accumulator0, P), Rescale(accumulator1, P)];
        }

        def main() {
            var ct_a0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("temp/ct_a0.npz",Q);
            var ct_a1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("temp/ct_a1.npz",Q);

            var ct_b0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("temp/ct_b0.npz",Q);
            var ct_b1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("temp/ct_b1.npz",Q);

            var prod0 = ct_a0 * ct_b0;
            var prod1 = ct_a0 * ct_b1 + ct_a1 * ct_b0;
            var prod2 = ct_a1 * ct_b1;

            var relin_d0_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d0_0.npz",QP);
            var relin_d0_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d0_1.npz",QP);
            var relin_d1_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d1_0.npz",QP);
            var relin_d1_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d1_1.npz",QP);
            var relin_d2_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d2_0.npz",QP);
            var relin_d2_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d2_1.npz",QP);
            var relin_d3_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d3_0.npz",QP);
            var relin_d3_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d3_1.npz",QP);
            var relin_d4_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d4_0.npz",QP);
            var relin_d4_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("temp/relin_d4_1.npz",QP);

            var ks = KeySwitch(prod2, relin_d0_0, relin_d0_1, relin_d1_0, relin_d1_1, relin_d2_0, relin_d2_1, relin_d3_0, relin_d3_1, relin_d4_0, relin_d4_1);
            var ct_res_0 = prod0 + get(ks, 0);
            var ct_res_1 = prod1 + get(ks, 1);
            
            var ct_res_0 = Rescale(ct_res_0,[0x7FFFE101]);
            var ct_res_1 = Rescale(ct_res_1,[0x7FFFE101]);
            
            write_mrp_u32_1024_Q(ct_res_0,"temp/ct_res_0.npz");
            write_mrp_u32_1024_Q(ct_res_1,"temp/ct_res_1.npz");
        }
        """
        ).program
        global_env = default_global()
        global_env.update(eval_globals(prog))
        eval_main(prog, global_env)
        ct_res_0 = load_mrp("temp/ct_res_0.npz")
        ct_res_1 = load_mrp("temp/ct_res_1.npz")
        decrypted_msg = ctx.decrypt_msg(Ciphertext(DEFAULT_SCALE, [ct_res_0, ct_res_1]))
        np.testing.assert_allclose(decrypted_msg, msg1 * msg2, rtol=1e-3, atol=1e-3)
    finally:
        shutil.rmtree("temp")
