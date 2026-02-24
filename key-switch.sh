#!/bin/bash
set -e

# Simulate the key_switch.fhetch program using CLI tools

mkdir -p temp

# Step 1: Generate keys (sk + relin keys)
echo "==> Generating keys..."
uv run fhetch-keygen params_65536.json -o temp/keys

# Step 2: Generate two random messages
echo "==> Generating random messages..."
uv run python scripts/generate-random-msg.py --output temp/msg1.json --size 32768
uv run python scripts/generate-random-msg.py --output temp/msg2.json --size 32768

# Step 3: Encrypt both messages
echo "==> Encrypting msg1..."
uv run fhetch-encrypt params_65536.json temp/keys/sk.npy temp/msg1.json -o temp/ct_a

echo "==> Encrypting msg2..."
uv run fhetch-encrypt params_65536.json temp/keys/sk.npy temp/msg2.json -o temp/ct_b

# Step 4: Run the fhetch key-switch program (multiply + relinearize + rescale)
mkdir -p temp/ct_res
echo "==> Running fhetch key_switch..."
uv run fhetch examples/key_switch.fhetch

# Step 5: Decrypt the result
echo "==> Decrypting result..."
uv run fhetch-decrypt params_65536.json temp/keys/sk.npy temp/ct_res -o temp/decrypted.npy

# Step 6: Verify the result
echo "==> Verifying result..."
uv run python scripts/verify_results.py temp/msg1.json temp/msg2.json temp/decrypted.npy

# Cleanup
echo ""
read -n 1 -s -r -p "To delete temp folder press any key..."
echo ""
rm -rf temp
echo "==> Cleaned up temp folder."
