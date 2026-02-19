# Key generation functions
import __future__ 
from math import prod
import math
from random import randint, shuffle, randbytes

import numpy as np

from client.crypto import Parameters
from client.utils import find_psi, random_poly
from fhetch.data import MRP, Vector
from fhetch.ntt import ROOTS_UNITY



DEFAULT_SCALE = 2.0**29

Q=[0x7ffe0001, 0x7ff80001, 0x7fea0001, 0x7fd20001]
P=[0x7fb40001, 0x7f440001]
for q in Q + P:
    ROOTS_UNITY[(16, q)] = find_psi(q,16)


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
    d_sizes = [2,2]
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
        default="secret_key.npy",
        help="Output path for secret key (default: secret_key.npy)"
    )
    
    args = parser.parse_args()
     # Load and deserialize parameters
    with open(args.params_json, 'r') as f:
        params_dict = json.load(f)
    
    params = Parameters(**params_dict)
    # Generate secret key
    sk = gen_sk(params)
    #  # Create MRP sk from a vector based on the pt base
    qp = params.q+params.p
    sk_poly = MRP.from_coeffs(base = qp , coeffs=sk.value)
    
    ksk = gen_ksk(sk_poly,sk_poly*sk_poly,Q,P)
    
    
    # Save to disk
    np.save(args.output, sk.value)
    print(f"Secret key generated and saved to {args.output}")
    
if __name__ == "__main__":
    main()


