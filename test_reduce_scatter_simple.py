"""
Test reduce-scatter in isolation.
"""

import os
import torch
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.utils import setup_distributed, reduce_scatter_tensor, get_rank, get_world_size


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] Testing reduce-scatter")
    
    # Create test tensor
    # All ranks have the SAME full tensor for testing
    full_tensor = torch.arange(12, dtype=torch.float32).to(device)
    
    print(f"[Rank {rank}] Input (full): sum={full_tensor.sum().item()}")
    
    # Reduce-scatter
    shard_size = 12 // world_size
    output = torch.zeros(shard_size, device=device)
    
    print(f"[Rank {rank}] Calling reduce-scatter...")
    reduce_scatter_tensor(output, full_tensor)
    
    print(f"[Rank {rank}] Output (shard): sum={output.sum().item()}")
    print(f"[Rank {rank}] Output values: {output.cpu().tolist()}")
    
    # Expected: sum of all ranks' inputs, then each rank gets a shard
    # All ranks have same input (0..11), so after sum: (0..11) * world_size
    # Each rank's shard should be: subset of (0..11) * world_size
    expected_total = torch.arange(12).float() * world_size
    shard_start = rank * shard_size
    shard_end = shard_start + shard_size
    expected_shard = expected_total[shard_start:shard_end]
    
    print(f"[Rank {rank}] Expected: {expected_shard.tolist()}")
    
    diff = (output.cpu() - expected_shard).abs().max()
    print(f"[Rank {rank}] Max diff: {diff:.2e}")
    
    if diff < 1e-6:
        print(f"[Rank {rank}] ✅ CORRECT")
    else:
        print(f"[Rank {rank}] ✗ WRONG")
    
    print(f"[Rank {rank}] Done")
    sys.exit(0)


if __name__ == "__main__":
    main()

