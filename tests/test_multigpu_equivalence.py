"""
FINAL STRICT TEST: Verify 1, 2, 4, 8 GPUs produce same final parameters.

All ranks use SAME data, SAME initialization.
After one training step, all should have SAME final parameters.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.flat_param import flatten_module_params
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, get_rank, get_world_size


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3, bias=False)
    
    def forward(self, x):
        return self.fc(x)


def train_one_step():
    """Train for one step and return final parameters."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        setup_distributed(rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # SAME initialization
    torch.manual_seed(42)
    model = SimpleModel()
    if world_size > 1:
        model = model.to(device)
    
    # Flatten and register hooks
    flat_param = flatten_module_params(model, rank=rank, world_size=world_size)
    register_forward_hooks(model, flat_param, reshard_after_forward=False)
    register_backward_hooks(model, flat_param, reshard_after_forward=False)
    
    # Optimizer
    optimizer = FSDPOptimizer(
        [flat_param],
        optimizer_cls=torch.optim.SGD,
        lr=0.1
    )
    
    # SAME data
    torch.manual_seed(100)
    x = torch.randn(2, 4)
    y = torch.randn(2, 3)
    if world_size > 1:
        x = x.to(device)
        y = y.to(device)
    
    # Train
    optimizer.zero_grad()
    out = model(x)
    loss = nn.MSELoss()(out, y)
    loss.backward()
    optimizer.step()
    
    # Get final parameters
    flat_param.all_gather()
    final_param = flat_param.full_param.detach().cpu().clone()
    final_loss = loss.item()
    
    if rank == 0:
        print(f"[{world_size} GPU] Loss: {final_loss:.10f}, Param sum: {final_param.sum().item():.10f}")
        torch.save({'param': final_param, 'loss': final_loss}, f"/tmp/strict_{world_size}gpu.pt")
    
    if world_size > 1:
        # Don't call cleanup to avoid hang
        pass
    
    sys.exit(0)


if __name__ == "__main__":
    train_one_step()

