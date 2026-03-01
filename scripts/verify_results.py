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
    print("PASS: Decrypted result matches msg1 * msg2\n")
    slots = range(4)
    col_w = 14
    rows = {
        "Message 1":     [f"{msg1[i]:.4f}"       for i in slots],
        "Message 2":     [f"{msg2[i]:.4f}"       for i in slots],
        "Expected":  [f"{expected[i]:.6f}"   for i in slots],
        "Decrypted": [f"{decrypted[i].real:.6f}" for i in slots],
        "Error":     [f"{abs(decrypted[i].real - expected[i]):.2e}" for i in slots],
    }
    header = "  ".join(["".rjust(col_w)] + [f"Slot {i}".rjust(col_w) for i in slots])
    print(header)
    print("  ".join("-" * col_w for _ in range(len(slots) + 1)))
    for label, values in rows.items():
        print("  ".join([label.rjust(col_w)] + [v.rjust(col_w) for v in values]))
    print()



if __name__ == "__main__":
    main()
