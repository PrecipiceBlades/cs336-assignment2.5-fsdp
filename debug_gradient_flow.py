"""
Debug gradient flow to find where the bug is.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.flat_param import FlatParameter, flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks, reduce_scatter_grads
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank, get_world_size


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 3))
    
    def forward(self, x):
        return x @ self.weight.t()


def test_single_gpu():
    print("="*60)
    print("SINGLE GPU")
    print("="*60)
    
    torch.manual_seed(42)
    
    module = SimpleLinear()
    flat_param = flatten_module_params(module, rank=0, world_size=1)
    
    print(f"\nInitial param: {flat_param.data.flatten().tolist()}")
    print(f"Param sum: {flat_param.data.sum().item():.10f}")
    
    # Register hooks
    register_forward_hooks(module, flat_param, reshard_after_forward=False)
    register_backward_hooks(module, flat_param, reshard_after_forward=False)
    
    # Data
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    
    # Forward
    out = module(x)
    print(f"\nOutput: {out}")
    print(f"Output sum: {out.sum().item():.10f}")
    
    # Loss
    loss = nn.MSELoss()(out, y)
    print(f"Loss: {loss.item():.10f}")
    
    # Backward
    loss.backward()
    
    print(f"\nAfter backward:")
    print(f"module.weight.grad: {module.weight.grad}")
    print(f"module.weight.grad sum: {module.weight.grad.sum().item():.10f}")
    
    print(f"\nflat_param.grad: {flat_param.grad}")
    print(f"flat_param.grad sum: {flat_param.grad.sum().item():.10f}")
    
    return flat_param.grad.clone(), loss.item()


def test_multi_gpu():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*60)
        print(f"MULTI-GPU (world_size={world_size})")
        print("="*60)
    
    torch.manual_seed(42)
    
    module = SimpleLinear().to(device)
    flat_param = flatten_module_params(module, rank=rank, world_size=world_size)
    
    if rank == 0:
        # All-gather to see full initial params
        flat_param.all_gather()
        print(f"\nInitial full param sum: {flat_param.full_param.sum().item():.10f}")
        flat_param.reshard()
    
    # Print each rank's shard
    print(f"[Rank {rank}] Local shard sum: {flat_param.data.sum().item():.10f}")
    
    # Register hooks
    register_forward_hooks(module, flat_param, reshard_after_forward=False)
    register_backward_hooks(module, flat_param, reshard_after_forward=False)
    
    # SAME data for all ranks
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(device)
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).to(device)
    
    # Check parameters before forward
    print(f"\n[Rank {rank}] Before forward, module.weight:")
    print(f"[Rank {rank}]   Shape: {module.weight.shape}")
    print(f"[Rank {rank}]   Sum: {module.weight.sum().item():.10f}")
    print(f"[Rank {rank}]   First row: {module.weight[0].tolist()}")
    
    # Forward
    out = module(x)
    
    if rank == 0:
        print(f"\nOutput: {out}")
        print(f"Output sum: {out.sum().item():.10f}")
    
    # Loss
    loss = nn.MSELoss()(out, y)
    
    if rank == 0:
        print(f"Loss: {loss.item():.10f}")
    
    # Backward
    if rank == 0:
        print(f"\nBefore backward...")
    
    loss.backward()
    
    # Print gradient info from each rank (without barrier to avoid timeout)
    print(f"\n[Rank {rank}] After backward:")
    if module.weight.grad is not None:
        print(f"  module.weight.grad sum: {module.weight.grad.sum().item():.10f}")
    
    if flat_param.grad is not None:
        print(f"  flat_param.grad sum: {flat_param.grad.sum().item():.10f}")
        print(f"  flat_param.grad numel: {flat_param.grad.numel()}")
    else:
        print(f"  flat_param.grad: None")
    
    print(f"[Rank {rank}] Exiting without cleanup...")
    import sys
    sys.exit(0)


if __name__ == "__main__":
    if "RANK" in os.environ:
        test_multi_gpu()
    else:
        grad, loss = test_single_gpu()
        print(f"\nFinal: grad_sum={grad.sum():.10f}, loss={loss:.10f}")

