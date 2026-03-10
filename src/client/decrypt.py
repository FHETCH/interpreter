import numpy as np
from client.context import CryptoContext
from client.crypto import Parameters
from client.utils import find_psi
from client import serialization
from fhetch.data import Vector
from fhetch.ntt import ROOTS_UNITY


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Decrypt FHE ciphertext")
    parser.add_argument("params_json", type=Path, help="Path to parameters JSON file")
    parser.add_argument("sk", type=Path, help="Path to secret key")
    parser.add_argument(
        "ciphertext",
        type=Path,
        help="Path to ciphertext directory (containing ct0.npz and ct1.npz)",
    )
    parser.add_argument("--scale", type=float, help="Scaling factor to use during decoding")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="decrypted_msg.npy",
        help="Output path for decrypted message (default: decrypted_msg.npy)",
    )

    args = parser.parse_args()

    # Load and deserialize parameters
    with open(args.params_json, "r") as f:
        params_dict = json.load(f)

    params = Parameters(**params_dict)
    for q in params.moduli:
        ROOTS_UNITY[(params.degree, q)] = find_psi(q, params.degree)

    # Load secret key
    sk = np.load(args.sk)
    ctx = CryptoContext(params, Vector(sk))

    # Load ciphertext
    ct0 = serialization.load_mrp(args.ciphertext / "ct0.npz")
    ct1 = serialization.load_mrp(args.ciphertext / "ct1.npz")

    # Decrypt
    decrypted_msg = ctx.decrypt_msg([ct0, ct1], args.scale)

    # Save to disk as text file with numpy array
    np.save(args.output, decrypted_msg)

    print(f"Message decrypted and saved to {args.output}")


if __name__ == "__main__":
    main()
