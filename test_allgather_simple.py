"""
Test all-gather correctness in the simplest possible way.
"""

import os
import torch
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.flat_param import FlatParameter
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank


def test_single():
    print("Single GPU:")
    p = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    fp = FlatParameter([p], rank=0, world_size=1)
    
    print(f"  Local shard: {fp.data.tolist()}")
    print(f"  Sum: {fp.data.sum().item()}")
    
    full = fp.all_gather()
    print(f"  After all_gather: {full.tolist()}")
    print(f"  Sum: {full.sum().item()}")
    print(f"  Expected sum: {sum(range(10))}")


def test_multi():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"\n[Rank {rank}] Multi-GPU (world_size={world_size}):")
    
    # Create same parameter on all ranks
    p = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(device))
    fp = FlatParameter([p], rank=rank, world_size=world_size)
    
    print(f"[Rank {rank}] Local shard: {fp.data.cpu().tolist()}")
    print(f"[Rank {rank}] Shard sum: {fp.data.sum().item()}")
    print(f"[Rank {rank}] Shard range: [{fp._shard_offset}:{fp._shard_offset + fp._shard_numel}]")
    
    # All-gather
    full = fp.all_gather()
    
    if rank == 0:
        print(f"\n[Rank 0] After all_gather: {full.cpu().tolist()}")
        print(f"[Rank 0] Full param sum: {full.sum().item()}")
        print(f"[Rank 0] Expected sum: {sum(range(10))}")
        
        if abs(full.sum().item() - sum(range(10))) < 1e-6:
            print(f"[Rank 0] ✅ All-gather CORRECT")
        else:
            print(f"[Rank 0] ✗ All-gather WRONG!")
    
    cleanup_distributed()


if __name__ == "__main__":
    if "RANK" in os.environ:
        test_multi()
    else:
        test_single()

