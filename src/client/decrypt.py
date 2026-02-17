import numpy as np
from client.context import CryptoContext
from client.crypto import Parameters
from client import serialization
from fhetch.data import Vector
from fhetch.ntt import ROOTS_UNITY

DEFAULT_SCALE = 2.0**29

Q = [0x10001, 0xC0001]
P = 0x120001
PSI = [2, 0xA27C, 0x31779]
for q, psi in zip(Q + [P], PSI):
    ROOTS_UNITY[(16, q)] = psi


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Decrypt FHE ciphertext")
    parser.add_argument(
        "params_json",
        type=Path,
        help="Path to parameters JSON file"
    )
    parser.add_argument(
        "sk",
        type=Path,
        help="Path to secret key"
    )
    parser.add_argument(
        "ciphertext",
        type=Path,
        help="Path to ciphertext file (.npz)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="decrypted_msg.txt",
        help="Output path for decrypted message (default: decrypted_msg.txt)"
    )
    
    args = parser.parse_args()
    
    # Load and deserialize parameters
    with open(args.params_json, 'r') as f:
        params_dict = json.load(f)
    
    params = Parameters(**params_dict)
    
    # Load secret key
    sk = np.load(args.sk)
    ctx = CryptoContext(params, Vector(sk))
    
    # Load ciphertext using unified serialization
    ct = serialization.load_ciphertext(args.ciphertext)
    
    # Decrypt
    decrypted_msg = ctx.decrypt_msg(ct)
    
    # Save to disk as text file with numpy array
    np.savetxt(args.output, decrypted_msg, fmt='%.6f')
    
    print(f"Message decrypted and saved to {args.output}")


if __name__ == "__main__":
    main()
