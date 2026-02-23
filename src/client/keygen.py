# Key generation functions
import __future__ 
from math import prod
import math
from random import randint, shuffle, randbytes

import numpy as np

from client.crypto import Parameters
from client.serialization import save_mrp
from client.utils import find_psi, random_poly
from fhetch.data import MRP, Vector
from fhetch.ntt import ROOTS_UNITY



DEFAULT_SCALE = 2.0**29


def gen_noise(moduli: list[int], degree: int, sigma=3.2):
    coeffs = []
    rng = np.random.default_rng()
    while len(coeffs) < degree:
        coeffInt = round(rng.normal(loc=0, scale=1, size=1)[0] * sigma)
        if abs(coeffInt) <= 6 * sigma:
            coeffs.append(coeffInt)
    return MRP.from_coeffs(moduli, coeffs)


def gen_sk(params:Parameters)->Vector:
    hw = params.h
    degree = params.degree
    if hw > degree:
        hw = degree // 2
    num_pos = randint(0, hw)
    num_neg = hw - num_pos
    coeffs = ([1] * num_pos) + ([-1] * num_neg) + ([0] * (degree - hw))
    shuffle(coeffs)
    return Vector(np.array(coeffs))

def gen_ksk(
    old_key: MRP,
    new_key: MRP,
    q: list[int],
    p: list[int],
):
    degree = old_key.degree()
    d_num = math.ceil(len(q) / len(p))
    P = prod(p)
    base = q + p

    a = [random_poly(base, degree) for _ in range(d_num)]
    d_sizes = [len(p)] * (d_num - 1) + [len(q) - len(p) * (d_num - 1)]
    powers = _powers(old_key, d_sizes)
    return [
        (b.muls(P) - a * new_key + gen_noise(base, degree), a)
        for b, a in zip(powers, a)
    ]

def _powers(poly: MRP, digit_sizes: list[int])->list[MRP]:
    moduli = list(poly.values.keys())
    QP = prod(moduli)
    idx = 0
    res = []
    for size in digit_sizes:
        q_hat = prod(moduli[idx:idx+size])
        idx += size
        # This is the same as clearing all the residues that are not in this digit
        res.append((poly.muls(QP // q_hat)).muls(pow(QP // q_hat, -1, q_hat)))
    return res


def gen_relin_key(sk: Vector, q: list[int], p: list[int]):
    """Generate a relinearization key from a secret key."""
    qp = q + p
    sk_poly = MRP.from_coeffs(base=qp, coeffs=sk.value)
    return gen_ksk(sk_poly * sk_poly, sk_poly, q, p)

      
def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate FHE secret key")
    parser.add_argument(
        "params_json",
        type=Path,
        help="Path to parameters JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="keys",
        help="Output folder for keys (default: keys)"
    )
    
    args = parser.parse_args()
     # Load and deserialize parameters
    with open(args.params_json, 'r') as f:
        params_dict = json.load(f)

    params = Parameters(**params_dict)
    for q in params.moduli:
        ROOTS_UNITY[(params.degree, q)] = find_psi(q, params.degree)

    # Generate secret key
    sk = gen_sk(params)
    # Save to disk
    np.save(args.output / "sk.npy", sk.value)
    
    relin_key = gen_relin_key(sk, params.q, params.p)

    args.output.mkdir(parents=True, exist_ok=True)
    for i, (ksk_0, ksk_1) in enumerate(relin_key):
        save_mrp(ksk_0, args.output / f"relin_d{i}_0.npz")
        save_mrp(ksk_1, args.output / f"relin_d{i}_1.npz")

    print(f"Secret key generated and saved to {args.output}")
    
if __name__ == "__main__":
    main()


