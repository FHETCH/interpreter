#!/bin/bash
set -e

# RING_DIM=65536
# MSG_SIZE=32768

RING_DIM=1024
MSG_SIZE=512

# Simulate the key_switch.fhetch program using CLI tools

step_start() { _STEP_T=$SECONDS; }
step_end()   { echo "    (took $(( SECONDS - _STEP_T ))s)"; }

mkdir -p temp

# Step 1: Generate keys (sk + relin keys)
echo "==> Generating keys..."
step_start
uv run fhetch-keygen params_${RING_DIM}.json -o temp/keys
step_end

# Step 2: Generate two random messages
echo "==> Generating random messages..."
step_start
uv run python scripts/generate-random-msg.py --output temp/msg1.json --size $MSG_SIZE
uv run python scripts/generate-random-msg.py --output temp/msg2.json --size $MSG_SIZE
step_end

# Step 3: Encrypt both messages
echo "==> Encrypting msg1..."
step_start
uv run fhetch-encrypt params_${RING_DIM}.json temp/keys/sk.npy temp/msg1.json -o temp/ct_a
step_end

echo "==> Encrypting msg2..."
step_start
uv run fhetch-encrypt params_${RING_DIM}.json temp/keys/sk.npy temp/msg2.json -o temp/ct_b
step_end

#Step 4: Run the fhetch key-switch program (multiply + relinearize + rescale)
mkdir -p temp/ct_res
echo "==> Running fhetch key_switch..."
step_start
uv run fhetch examples/key_switch_${RING_DIM}.fhetch
step_end

# Step 5: Decrypt the result
echo "==> Decrypting result..."
step_start
uv run fhetch-decrypt params_${RING_DIM}.json temp/keys/sk.npy temp/ct_res -o temp/decrypted.npy
step_end

# Step 6: Verify the result
echo "==> Verifying result..."
step_start
uv run python scripts/verify_results.py temp/msg1.json temp/msg2.json temp/decrypted.npy
step_end

# Cleanup
echo ""
read -n 1 -s -r -p "To delete temp folder press any key..."
echo ""
rm -rf temp
echo "==> Cleaned up temp folder."
