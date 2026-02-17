import numpy as np
from client.context import CryptoContext
from client.crypto import Ciphertext, Parameters
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
    
    # Load ciphertext
    ct_data = np.load(args.ciphertext)
    moduli = ct_data['moduli']
    scale = ct_data['scale'].item() if 'scale' in ct_data else 2**30
    
    # Reconstruct MRP objects from limbs
    from fhetch.data import MRP
    ct_0_values = {int(moduli[i]): Vector(ct_data[f'ct_0_limb_{i}']) for i in range(len(moduli))}
    ct_1_values = {int(moduli[i]): Vector(ct_data[f'ct_1_limb_{i}']) for i in range(len(moduli))}
    
    ct_0 = MRP(ct_0_values)
    ct_1 = MRP(ct_1_values)
    ct = Ciphertext(scale=scale, polynomials=[ct_0, ct_1])
    
    # Decrypt
    decrypted_msg = ctx.decrypt_msg(ct)
    
    # Save to disk as text file with numpy array
    np.savetxt(args.output, decrypted_msg, fmt='%.6f')
    
    print(f"Message decrypted and saved to {args.output}")


if __name__ == "__main__":
    main()
