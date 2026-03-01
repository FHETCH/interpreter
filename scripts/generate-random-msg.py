import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate a random message for FHE encryption")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--size", type=int, default=512, help="Number of message slots (default: 512)")
    parser.add_argument("--name", type=str, default="MSG", help="Name of the message to be printed")
    args = parser.parse_args()

    msg = (np.random.uniform(0, 10, size=args.size) * 100).round() / 100
    msg = msg.tolist()

    with open(args.output, "w") as f:
        json.dump(msg, f)
    print(f"{args.name}: [{', '.join(str(x) for x in msg[:4])}...]")


if __name__ == "__main__":
    main()
