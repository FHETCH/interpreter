import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify FHE multiplication result")
    parser.add_argument("msg1", help="Path to first message JSON")
    parser.add_argument("msg2", help="Path to second message JSON")
    parser.add_argument("decrypted", help="Path to decrypted result file")
    args = parser.parse_args()

    with open(args.msg1) as f:
        msg1 = np.array(json.load(f))
    with open(args.msg2) as f:
        msg2 = np.array(json.load(f))
    decrypted = np.load(args.decrypted)

    expected = msg1 * msg2

    np.testing.assert_allclose(decrypted, expected, rtol=1e-3, atol=1e-3)
    print("PASS: Decrypted result matches msg1 * msg2")


if __name__ == "__main__":
    main()
