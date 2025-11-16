"""
Test forward only - no backward to isolate the bug.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.forward_pass import all_gather_params
from fsdp.utils import setup_distributed, get_rank


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # 4x3 weight matrix, initialized to all ones
        self.weight = nn.Parameter(torch.ones(4, 3))
    
    def forward(self, x):
        # x: [2, 3], weight.t(): [3, 4] → output: [2, 4]
        return x @ self.weight.t()


def main():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        setup_distributed(rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cpu")
    
    print(f"[Rank {rank}/{world_size}] Starting test")
    
    # Create module
    torch.manual_seed(42)
    module = SimpleLinear()
    if world_size > 1:
        module = module.to(device)
    
    # Create FlatParameter
    flat_param = flatten_module_params(module, rank=rank, world_size=world_size)
    
    print(f"[Rank {rank}] FlatParam created:")
    print(f"  total_numel: {flat_param._total_numel}")
    print(f"  padded_numel: {flat_param._padded_total_numel}")
    print(f"  shard numel: {flat_param.data.numel()}")
    print(f"  shard sum: {flat_param.data.sum().item():.6f}")
    
    # All-gather
    print(f"\n[Rank {rank}] Calling all_gather...")
    full_param = flat_param.all_gather()
    
    print(f"[Rank {rank}] All-gather done:")
    print(f"  full_param numel: {full_param.numel()}")
    print(f"  full_param sum: {full_param.sum().item():.6f}")
    print(f"  Expected: 12.0")
    
    # Update views
    print(f"\n[Rank {rank}] Creating views and updating module...")
    views = flat_param.create_views()
    print(f"[Rank {rank}]   Created {len(views)} views")
    
    if len(views) > 0:
        print(f"[Rank {rank}]   View 0 shape: {views[0].shape}, sum: {views[0].sum().item():.6f}")
        module.weight.data = views[0]
        print(f"[Rank {rank}]   Updated module.weight")
    
    # Check module.weight
    print(f"\n[Rank {rank}] After update, module.weight:")
    print(f"  Shape: {module.weight.shape}")
    print(f"  Sum: {module.weight.sum().item():.6f}")
    print(f"  Data: {module.weight.flatten().tolist()}")
    
    # Forward pass
    print(f"\n[Rank {rank}] Running forward...")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    if world_size > 1:
        x = x.to(device)
    
    out = module(x)
    print(f"[Rank {rank}] Output: {out.cpu().tolist()}")
    print(f"[Rank {rank}] Expected: [[6, 6, 6, 6], [15, 15, 15, 15]]")
    
    print(f"\n[Rank {rank}] Test completed")


if __name__ == "__main__":
    main()

