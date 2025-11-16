#!/bin/bash
set -e

cd /root/cs336-assignment2.5-fsdp
rm -f /tmp/strict_*gpu.pt

echo "Testing 1 GPU..."
uv run python test_multigpu_strict_final.py

echo "Testing 2 GPUs..."
timeout 10 uv run torchrun --nproc_per_node=2 test_multigpu_strict_final.py 2>&1 | grep -v "W1116"

echo "Testing 4 GPUs..."
timeout 10 uv run torchrun --nproc_per_node=4 test_multigpu_strict_final.py 2>&1 | grep -v "W1116"

echo "Testing 8 GPUs..."
timeout 10 uv run torchrun --nproc_per_node=8 test_multigpu_strict_final.py 2>&1 | grep -v "W1116"

echo ""
echo "Comparing results..."
python3 << 'EOF'
import torch

results = {}
for ws in [1, 2, 4, 8]:
    try:
        results[ws] = torch.load(f"/tmp/strict_{ws}gpu.pt")
    except Exception as e:
        print(f"Failed to load {ws} GPU results: {e}")
        exit(1)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

for ws in [1, 2, 4, 8]:
    r = results[ws]
    print(f"{ws} GPU:  Loss={r['loss']:.10f}, Param_sum={r['param'].sum().item():.10f}")

print("\n" + "="*70)
print("COMPARISON vs 1 GPU")
print("="*70)

baseline = results[1]['param']
max_diff = 0.0

for ws in [2, 4, 8]:
    diff = (baseline - results[ws]['param']).abs().max().item()
    max_diff = max(max_diff, diff)
    print(f"{ws} GPU: max_diff = {diff:.2e}")

print(f"\nOverall max diff: {max_diff:.2e}")

if max_diff < 1e-6:
    print("✅ ALL GPU COUNTS EXACTLY EQUIVALENT!")
    exit(0)
else:
    print(f"✗ FAILED - diff {max_diff:.2e} exceeds 1e-6")
    exit(1)
EOF

