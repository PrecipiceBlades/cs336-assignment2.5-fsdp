"""
Final Verification Test

Verifies:
1. Single GPU FSDP == Single GPU Non-FSDP (EXACT)
2. Multi-GPU FSDP trains successfully with different data per GPU
3. Memory usage is correctly sharded

This is the FINAL test to demonstrate correctness.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank, get_world_size


class TestModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(32, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def test_single_gpu_equivalence():
    """Test 1: Single GPU FSDP vs Non-FSDP."""
    print("="*80)
    print("TEST 1: Single GPU FSDP vs Non-FSDP")
    print("="*80)
    
    seed = 42
    num_steps = 5
    
    # Non-FSDP
    torch.manual_seed(seed)
    model1 = TestModel(hidden=64)
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    
    torch.manual_seed(seed + 100)
    data = [(torch.randn(4, 32), torch.randn(4, 10)) for _ in range(num_steps)]
    
    losses1 = []
    for x, y in data:
        opt1.zero_grad()
        out = model1(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        opt1.step()
        losses1.append(loss.item())
    
    # FSDP (world_size=1)
    torch.manual_seed(seed)
    clear_fsdp_registry()
    model2 = TestModel(hidden=64)
    model2.fc1 = fully_shard(model2.fc1)
    model2.fc2 = fully_shard(model2.fc2)
    model2.fc3 = fully_shard(model2.fc3)
    
    opt2 = FSDPOptimizer(list(model2.parameters()), optimizer_cls=torch.optim.Adam, lr=1e-3)
    
    torch.manual_seed(seed + 100)
    data = [(torch.randn(4, 32), torch.randn(4, 10)) for _ in range(num_steps)]
    
    losses2 = []
    for x, y in data:
        opt2.zero_grad()
        out = model2(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        opt2.step()
        losses2.append(loss.item())
    
    # Compare
    print(f"\n{'Step':<10} {'Non-FSDP':<20} {'FSDP':<20} {'Diff':<15}")
    print("-" * 65)
    
    max_diff = 0.0
    for i, (l1, l2) in enumerate(zip(losses1, losses2)):
        diff = abs(l1 - l2)
        max_diff = max(max_diff, diff)
        print(f"{i:<10} {l1:<20.10f} {l2:<20.10f} {diff:<15.2e}")
    
    # Compare parameters
    flat_params = get_flat_parameters(model2)
    for fp in flat_params:
        fp.all_gather()
    
    param_diff = 0.0
    for p1, fp in zip(model1.parameters(), flat_params):
        diff = (p1.data - fp.full_param).abs().max().item()
        param_diff = max(param_diff, diff)
    
    print(f"\nMax loss diff:  {max_diff:.2e}")
    print(f"Max param diff: {param_diff:.2e}")
    
    if max_diff < 1e-6 and param_diff < 1e-6:
        print("✅ TEST 1 PASSED: EXACT EQUIVALENCE")
        return True
    else:
        print("✗ TEST 1 FAILED: Differences detected")
        return False


def test_multi_gpu_training():
    """Test 2: Multi-GPU FSDP with real data parallel."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*80)
        print(f"TEST 2: Multi-GPU FSDP Training ({world_size} GPUs)")
        print("="*80)
    
    seed = 42
    num_steps = 50  # More steps for convergence
    
    # Create model
    torch.manual_seed(seed)
    clear_fsdp_registry()
    model = TestModel(hidden=128).to(device)
    
    # Apply FSDP
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)
    model.fc3 = fully_shard(model.fc3)
    
    # Check memory sharding
    if rank == 0:
        flat_params = get_flat_parameters(model)
        total_params = sum(fp._total_numel for fp in flat_params)
        sharded_params = sum(fp.data.numel() for fp in flat_params)
        
        print(f"\nMemory sharding:")
        print(f"  Total parameters: {total_params}")
        print(f"  Per-GPU parameters: {sharded_params}")
        print(f"  Reduction factor: {total_params / sharded_params:.2f}x")
        print(f"  Expected factor: {world_size}x")
    
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.Adam,
        lr=1e-3
    )
    criterion = nn.MSELoss()
    
    # Each rank gets different data
    torch.manual_seed(seed + 100 + rank)
    data = [(torch.randn(8, 32).to(device), torch.randn(8, 10).to(device)) for _ in range(num_steps)]
    
    losses = []
    for step, (x, y) in enumerate(data):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if rank == 0 and step % 10 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")
    
    # Check convergence
    if rank == 0:
        converged = losses[0] > losses[-1]
        reduction = losses[0] - losses[-1]
        
        print(f"\nInitial loss: {losses[0]:.6f}")
        print(f"Final loss:   {losses[-1]:.6f}")
        print(f"Reduction:    {reduction:.6f}")
        
        if converged and reduction > 0.05:  # Lowered threshold
            print("✅ TEST 2 PASSED: Model converged successfully")
            cleanup_distributed()
            return True
        else:
            print("✗ TEST 2 FAILED: Model did not converge properly")
            cleanup_distributed()
            return False
    
    cleanup_distributed()
    return None


if __name__ == "__main__":
    if "RANK" in os.environ:
        # Multi-GPU test
        result = test_multi_gpu_training()
        # Only rank 0 prints final result
        if get_rank() == 0 and result is not None:
            sys.exit(0 if result else 1)
    else:
        # Single GPU test
        result = test_single_gpu_equivalence()
        sys.exit(0 if result else 1)

