import pytest
import numpy as np
from client.context import CryptoContext, Parameters, decode, encode
from client.serialization import load_mrp, save_mrp
from client.keygen import gen_relin_key, gen_sk
from client.utils import find_psi
from fhetch import parser
from fhetch.env import default_global
from fhetch.eval import eval_globals, eval_main
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
    msg = np.random.randint(0, np.iinfo(np.int16).max, size=512)

    # Encrypt the message
    pt = encode(msg, ctx._params.scaling_factor(), ctx._params.moduli)

    # Decrypt the ciphertext
    dec_msg = decode(pt)

    # Check that the decrypted message matches the original
    # Use allclose for floating point comparison with tolerance
    np.testing.assert_allclose(dec_msg, msg, rtol=1e-3, atol=1e-3)


def test_encrypt_decrypt(ctx: CryptoContext):
    """Test that encrypting and decrypting a message returns the original message."""
    # Create a test message
    msg = np.random.randint(0, np.iinfo(np.int16).max, size=512)

    # Encrypt the message
    ciphertext = ctx.encrypt_msg(msg)

    # Decrypt the ciphertext
    decrypted_msg = ctx.decrypt_msg(ciphertext)

    # Check that the decrypted message matches the original
    # Use allclose for floating point comparison with tolerance
    np.testing.assert_allclose(decrypted_msg, msg, rtol=1e-3, atol=1e-3)


def test_add(ctx: CryptoContext, tmp_path):
    msg = np.random.randint(0, np.iinfo(np.int16).max, size=512)

    ciphertext = ctx.encrypt_msg(list(msg), ctx._params.scaling_factor())

    save_mrp(ciphertext[0], str(tmp_path / "ct_0.npz"))
    save_mrp(ciphertext[1], str(tmp_path / "ct_1.npz"))

    prog = parser.Program.parse_string(
        f"""
    {FHETCH_PRIMES}
    def main() {{
        var ct_0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("{tmp_path}/ct_0.npz",Q);
        var ct_1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("{tmp_path}/ct_1.npz",Q);
        var s0 = ct_0 + ct_0;
        var s1 = ct_1 + ct_1;
        write_mrp_u32_1024_Q(s0,"{tmp_path}/ct_res_0.npz");
        write_mrp_u32_1024_Q(s1,"{tmp_path}/ct_res_1.npz");
    }}
    """
    ).program
    global_env = default_global()
    global_env.update(eval_globals(prog))
    eval_main(prog, global_env)
    ct_res_0 = load_mrp(str(tmp_path / "ct_res_0.npz"))
    ct_res_1 = load_mrp(str(tmp_path / "ct_res_1.npz"))
    decrypted_msg = ctx.decrypt_msg([ct_res_0, ct_res_1])
    np.testing.assert_allclose(decrypted_msg, msg + msg, rtol=1e-3, atol=1e-3)


def test_mult(ctx: CryptoContext, tmp_path):
    msg1 = np.random.randint(0, np.iinfo(np.int16).max, size=512)
    msg2 = np.random.randint(0, np.iinfo(np.int16).max, size=512)

    ct_a = ctx.encrypt_msg(list(msg1))
    ct_b = ctx.encrypt_msg(list(msg2))

    save_mrp(ct_a[0], str(tmp_path / "ct_a0.npz"))
    save_mrp(ct_a[1], str(tmp_path / "ct_a1.npz"))

    save_mrp(ct_b[0], str(tmp_path / "ct_b0.npz"))
    save_mrp(ct_b[1], str(tmp_path / "ct_b1.npz"))

    relin_key = gen_relin_key(ctx._sk, Q, P)

    for i, (ksk_0, ksk_1) in enumerate(relin_key):
        save_mrp(ksk_0, str(tmp_path / f"relin_d{i}_0.npz"))
        save_mrp(ksk_1, str(tmp_path / f"relin_d{i}_1.npz"))

    prog = parser.Program.parse_string(
        f"""
    {FHETCH_PRIMES}
    def KeySwitch(poly: MRP<u32, 1024, Q>, k00, k01, k10, k11, k20, k21, k30, k31, k40, k41) {{ 
        var decomposed0 = BaseExtend(poly, Digit0, QP);
        var decomposed1 = BaseExtend(poly, Digit1, QP);
        var decomposed2 = BaseExtend(poly, Digit2, QP);
        var decomposed3 = BaseExtend(poly, Digit3, QP);
        var decomposed4 = BaseExtend(poly, Digit4, QP);

        var accumulator0 = decomposed0 * k00 + decomposed1 * k10 + decomposed2 * k20 + decomposed3 * k30 + decomposed4 * k40;
        var accumulator1 = decomposed0 * k01 + decomposed1 * k11 + decomposed2 * k21 + decomposed3 * k31 + decomposed4 * k41;

        return [Rescale(accumulator0, P), Rescale(accumulator1, P)];
    }}

    def main() {{
        var ct_a0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("{tmp_path}/ct_a0.npz",Q);
        var ct_a1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("{tmp_path}/ct_a1.npz",Q);

        var ct_b0: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("{tmp_path}/ct_b0.npz",Q);
        var ct_b1: MRP<u32, 1024, Q> = read_mrp_u32_1024_Q("{tmp_path}/ct_b1.npz",Q);

        var prod0 = ct_a0 * ct_b0;
        var prod1 = ct_a0 * ct_b1 + ct_a1 * ct_b0;
        var prod2 = ct_a1 * ct_b1;

        var relin_d0_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d0_0.npz",QP);
        var relin_d0_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d0_1.npz",QP);
        var relin_d1_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d1_0.npz",QP);
        var relin_d1_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d1_1.npz",QP);
        var relin_d2_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d2_0.npz",QP);
        var relin_d2_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d2_1.npz",QP);
        var relin_d3_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d3_0.npz",QP);
        var relin_d3_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d3_1.npz",QP);
        var relin_d4_0: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d4_0.npz",QP);
        var relin_d4_1: MRP<u32, 1024, QP> = read_mrp_u32_1024_Q("{tmp_path}/relin_d4_1.npz",QP);

        var ks = KeySwitch(prod2, relin_d0_0, relin_d0_1, relin_d1_0, relin_d1_1, relin_d2_0, relin_d2_1, relin_d3_0, relin_d3_1, relin_d4_0, relin_d4_1);
        var ct_res_0 = prod0 + get(ks, 0);
        var ct_res_1 = prod1 + get(ks, 1);
        
        var ct_res_0 = Rescale(ct_res_0,[0x7FF81001]);
        var ct_res_1 = Rescale(ct_res_1,[0x7FF81001]);
        
        write_mrp_u32_1024_Q(ct_res_0,"{tmp_path}/ct_res_0.npz");
        write_mrp_u32_1024_Q(ct_res_1,"{tmp_path}/ct_res_1.npz");
    }}
    """
    ).program
    global_env = default_global()
    global_env.update(eval_globals(prog))
    eval_main(prog, global_env)
    ct_res_0 = load_mrp(str(tmp_path / "ct_res_0.npz"))
    ct_res_1 = load_mrp(str(tmp_path / "ct_res_1.npz"))
    decrypted_msg = ctx.decrypt_msg([ct_res_0, ct_res_1])
    np.testing.assert_allclose(decrypted_msg, msg1 * msg2, rtol=1e-3, atol=1e-3)
