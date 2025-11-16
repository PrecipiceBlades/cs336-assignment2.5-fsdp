#!/bin/bash
set -e

echo "Running STRICT multi-GPU equivalence test"
echo "All GPUs will process SAME data to verify mathematical correctness"
echo ""

cd /root/cs336-assignment2.5-fsdp
rm -f /tmp/fsdp_strict_*gpu.pt

echo "1 GPU..."
uv run python test_strict_multigpu.py

echo ""
echo "2 GPUs..."
uv run torchrun --nproc_per_node=2 test_strict_multigpu.py

echo ""
echo "4 GPUs..."
uv run torchrun --nproc_per_node=4 test_strict_multigpu.py

echo ""
echo "8 GPUs..."
uv run torchrun --nproc_per_node=8 test_strict_multigpu.py

echo ""
echo "Comparing..."
uv run python test_strict_multigpu.py --compare

