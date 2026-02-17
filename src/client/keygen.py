# Key generation functions
import __future__ 
from math import prod
from random import randint, shuffle, randbytes

import numpy as np

from client.crypto import Parameters
from fhetch.data import MRP, Vector


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
    
    # Save to disk
    np.save(args.output, sk.value)
    print(f"Secret key generated and saved to {args.output}")
    
if __name__ == "__main__":
    main()


