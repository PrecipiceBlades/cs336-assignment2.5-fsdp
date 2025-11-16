"""
Isolated backward test - minimal reproduction.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.flat_param import flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.utils import setup_distributed, get_rank


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3))
    
    def forward(self, x):
        return x @ self.weight.t()


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        setup_distributed(rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    print(f"[Rank {rank}] Starting")
    
    # Create module
    torch.manual_seed(42)
    module = TinyLinear()
    if world_size > 1:
        module = module.to(device)
    
    # Flatten
    flat_param = flatten_module_params(module, rank=rank, world_size=world_size)
    print(f"[Rank {rank}] FlatParam: total={flat_param._total_numel}, shard={flat_param.data.numel()}")
    print(f"[Rank {rank}] Number of orig_params: {len(flat_param._orig_params)}")
    
    # Register hooks (NO reshard for simplicity)
    register_forward_hooks(module, flat_param, reshard_after_forward=False)
    register_backward_hooks(module, flat_param, reshard_after_forward=False)
    
    # Data
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = torch.tensor([[1.0, 0.0]])
    if world_size > 1:
        x = x.to(device)
        y = y.to(device)
    
    # Forward
    print(f"[Rank {rank}] Forward...")
    out = module(x)
    print(f"[Rank {rank}] Output: {out.cpu().tolist()}")
    
    # Loss
    loss = nn.MSELoss()(out, y)
    print(f"[Rank {rank}] Loss: {loss.item():.6f}")
    
    # Backward
    print(f"[Rank {rank}] Backward...")
    loss.backward()
    
    print(f"[Rank {rank}] Backward done!")
    if flat_param.grad is not None:
        print(f"[Rank {rank}] flat_param.grad: {flat_param.grad.cpu().tolist()}")
        print(f"[Rank {rank}] flat_param.grad sum: {flat_param.grad.sum().item()}")
    
    # Also check module.weight.grad
    if module.weight.grad is not None:
        print(f"[Rank {rank}] module.weight.grad sum: {module.weight.grad.sum().item()}")
    
    print(f"[Rank {rank}] SUCCESS")
    sys.exit(0)


if __name__ == "__main__":
    os.environ["DEBUG_GRADS"] = "1"
    main()

