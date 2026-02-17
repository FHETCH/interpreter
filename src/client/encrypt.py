import numpy as np
from client.context import CryptoContext
from client.crypto import Parameters
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

    parser = argparse.ArgumentParser(description="Encrypt FHE msg")
    parser.add_argument("params_json", type=Path, help="Path to parameters JSON file")
    parser.add_argument("sk", type=Path, help="Path to secret key")
    parser.add_argument("msg", type=Path, help="Path to message (text file with numpy array)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="ct.npz",
        help="Output path for Ciphertext (default: ct.npz)",
    )

    args = parser.parse_args()
    # Load and deserialize parameters
    with open(args.params_json, "r") as f:
        params_dict = json.load(f)
    
    # Load message as numpy array from text file
    msg = np.loadtxt(args.msg)

    params = Parameters(**params_dict)

    # load secret key
    sk = np.load(args.sk)
    assert len(sk) == 2 * len(msg)
    ctx = CryptoContext(params, Vector(sk))

    ct = ctx.encrypt_msg(msg, DEFAULT_SCALE)

    # Save to disk - save each RNS limb separately
    ct_0 = ct.polynomials[0]
    ct_1 = ct.polynomials[1]
    
    # Get moduli (should be the same for both polynomials)
    moduli = np.array(list(ct_0.values.keys()))
    
    # Save all limbs with their moduli
    save_dict = {'moduli': moduli, 'scale': ct.scale}
    for i, mod in enumerate(moduli):
        save_dict[f'ct_0_limb_{i}'] = ct_0.values[mod].value
        save_dict[f'ct_1_limb_{i}'] = ct_1.values[mod].value
    
    np.savez(args.output, **save_dict)
    print(f"Ciphertext generated and saved to {args.output}")


if __name__ == "__main__":
    main()
