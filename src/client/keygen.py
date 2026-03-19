# Key generation functions
from math import prod
import math
from random import randint, shuffle

import numpy as np

from client.crypto import Parameters
from client.serialization import save_mrp, save_ksk
from client.utils import find_psi, random_poly
from fhetch.data import MRP, Vector
from fhetch.ntt import ROOTS_UNITY


def gen_noise(moduli: list[int], degree: int, sigma=3.2):
    coeffs = []
    rng = np.random.default_rng()
    while len(coeffs) < degree:
        sample = rng.normal(scale=sigma, size=degree - len(coeffs)).round()
        good_idxs = np.abs(sample) <= 6 * sigma
        coeffs.extend(int(x) for x in sample[good_idxs])
    return MRP.from_coeffs(moduli, coeffs)

def negacyclic_automorphism(coeffs, g):
    """
    Applies the automorphism x -> x^g to a polynomial in Z[x]/(x^N + 1).
    coeffs: list or np.array of coefficients of length N
    g: the Galois element (e.g., pow(5, rot, 2*N))
    """
    N = len(coeffs)
    new_coeffs = np.zeros(N, dtype=coeffs.dtype)
    
    for i in range(N):
        # Calculate the new exponent: (original_exponent * g) % (2 * N)
        target_exp = (i * g) % (2 * N)
        
        if target_exp < N:
            # Standard position
            new_coeffs[target_exp] = coeffs[i]
        else:
            # Negacyclic wrap-around: x^N = -1, so x^(N+k) = -x^k
            new_coeffs[target_exp - N] = -coeffs[i]
            
    return new_coeffs

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
    zeros = Vector(np.zeros(degree, dtype=np.uint32))

    # For digit i, powers[i][q_j] == old_key[q_j] if q_j is in digit i, else 0.
    # we directly build the scaled MRP:
    # multiply only the digit's residues by P, zero-fill the rest.
    idx = 0
    result = []
    for size, a_poly in zip(d_sizes, a):
        digit_set = set(q[idx:idx + size])
        b_scaled = MRP({
            q_j: (old_key.values[q_j].mul(P ,q_j)) if q_j in digit_set else zeros
            for q_j in base
        })
        result.append((b_scaled - a_poly * new_key + gen_noise(base, degree), a_poly))
        idx += size
    return result


def gen_relin_key(sk: Vector, q: list[int], p: list[int]):
    """Generate a relinearization key from a secret key."""
    qp = q + p
    sk_poly = MRP.from_coeffs(base=qp, coeffs=sk.value)
    return gen_ksk(sk_poly * sk_poly, sk_poly, q, p)

def gen_rotation_key(sk:Vector,q: list[int], p: list[int],rot:int):
    qp = q + p
    # Convert to Polynomial format
    sk_poly = MRP.from_coeffs(base=qp, coeffs=sk.value)
    rotated_sk_poly = sk_poly.automorph(rot)
    # This is an encryption of the 'rotated' key under the 'original' key
    rot_key = gen_ksk(rotated_sk_poly, sk_poly, q, p)
    
    return rot_key
      
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
    relin_key = gen_relin_key(sk, params.q, params.p)
    rot_by_1 = gen_rotation_key(sk, params.q, params.p,1)
    
    # Save to disk
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "sk.npy", sk.value)

    save_ksk(relin_key, args.output, "relin")
    save_ksk(rot_by_1, args.output, "rot_by_1")

    print(f"Secret key generated and saved to {args.output}")
    
if __name__ == "__main__":
    main()


