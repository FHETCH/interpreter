import numpy as np
from client.context import CryptoContext
from client.crypto import Parameters
from client import serialization
from client.utils import find_psi
from fhetch.data import Vector
from fhetch.ntt import ROOTS_UNITY

def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Encrypt FHE msg")
    parser.add_argument("params_json", type=Path, help="Path to parameters JSON file")
    parser.add_argument("sk", type=Path, help="Path to secret key")
    parser.add_argument("msg", type=Path, help="Path to message (text file with numpy array)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="ct",
        help="Output directory for Ciphertext (default: ct)",
    )

    args = parser.parse_args()
    # Load and deserialize parameters
    with open(args.params_json, "r") as f:
        params_dict = json.load(f)
    
    # Load message as numpy array from text file
    msg = json.load(open(args.msg))

    params = Parameters(**params_dict)
    for q in params.moduli:
        ROOTS_UNITY[(params.degree, q)] = find_psi(q, params.degree)

    # load secret key
    sk = np.load(args.sk)
    assert len(sk) == 2 * len(msg)
    ctx = CryptoContext(params, Vector(sk))

    ct = ctx.encrypt_msg(msg)

    # Save to disk using unified serialization
    args.output.mkdir(parents=True, exist_ok=True)
    serialization.save_mrp(ct[0], args.output / "ct0.npz")
    serialization.save_mrp(ct[1], args.output / "ct1.npz")
    
    print(f"Ciphertext generated and saved to {args.output}/ct0.npz and {args.output}/ct1.npz")


if __name__ == "__main__":
    main()
