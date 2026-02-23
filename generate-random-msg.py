import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate a random message for FHE encryption")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--size", type=int, default=512, help="Number of message slots (default: 512)")
    args = parser.parse_args()

    msg = np.random.randint(0, np.iinfo(np.int16).max, size=args.size).tolist()

    with open(args.output, "w") as f:
        json.dump(msg, f)


if __name__ == "__main__":
    main()
