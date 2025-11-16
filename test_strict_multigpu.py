"""
STRICT Multi-GPU Equivalence Test

Verify that 1, 2, 4, 8 GPUs produce EXACTLY the same final parameters
when all ranks process the SAME data in the SAME order.

This is the TRUE test of mathematical correctness for FSDP sharding logic.
"""

import os
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/cs336-assignment2.5-fsdp')

from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)
    
    def forward(self, x):
        return self.fc(x)


def run_training(world_size_str):
    """Run training and save results."""
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
    
    # SAME initialization for all ranks and all world_sizes
    torch.manual_seed(42)
    clear_fsdp_registry()
    
    model = TinyModel()
    if world_size > 1:
        model = model.to(device)
    
    # Apply FSDP
    model.fc = fully_shard(model.fc)
    
    flat_params = get_flat_parameters(model)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training with world_size={world_size}")
        print(f"{'='*60}")
        for i, fp in enumerate(flat_params):
            print(f"FP{i}: total={fp._total_numel}, padded={fp._padded_total_numel}, shard={fp.data.numel()}")
    
    # Optimizer
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.SGD,
        lr=0.1
    )
    
    # SAME data for all ranks (critical!)
    torch.manual_seed(100)
    x = torch.randn(2, 8)
    y = torch.randn(2, 4)
    
    if world_size > 1:
        x = x.to(device)
        y = y.to(device)
    
    criterion = nn.MSELoss()
    
    # Single training step
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    
    if rank == 0:
        print(f"Before backward - Loss: {loss.item():.15f}")
    
    loss.backward()
    
    # Print gradient info
    if rank == 0:
        for i, fp in enumerate(flat_params):
            if fp.grad is not None:
                print(f"FP{i} grad sum: {fp.grad.sum().item():.15f}")
            else:
                print(f"FP{i} grad: None")
    
    optimizer.step()
    
    # Get final parameters
    if rank == 0:
        for fp in flat_params:
            fp.all_gather()
        
        final_params = [fp.full_param.detach().cpu().clone() for fp in flat_params]
        final_loss = loss.item()
        
        print(f"After step - Param sum: {sum(p.sum().item() for p in final_params):.15f}")
        
        # Save
        torch.save({
            'loss': final_loss,
            'params': final_params,
            'world_size': world_size
        }, f"/tmp/fsdp_strict_{world_size}gpu.pt")
        
        print(f"Saved to /tmp/fsdp_strict_{world_size}gpu.pt")
    
    if world_size > 1:
        cleanup_distributed()


def compare_all():
    """Compare results from all GPU counts."""
    results = {}
    for ws in [1, 2, 4, 8]:
        try:
            results[ws] = torch.load(f"/tmp/fsdp_strict_{ws}gpu.pt")
        except:
            print(f"⚠️  No results for {ws} GPU(s)")
            return False
    
    print("\n" + "="*80)
    print("STRICT EQUIVALENCE COMPARISON")
    print("="*80)
    
    # Compare losses
    print("\nLosses:")
    for ws in [1, 2, 4, 8]:
        print(f"  {ws} GPU(s): {results[ws]['loss']:.15f}")
    
    # Compare parameters
    print("\nParameter Comparisons (vs 1 GPU baseline):")
    baseline_params = results[1]['params']
    
    max_diff_overall = 0.0
    
    for ws in [2, 4, 8]:
        test_params = results[ws]['params']
        
        print(f"\n{ws} GPU(s):")
        for i, (bp, tp) in enumerate(zip(baseline_params, test_params)):
            diff = (bp - tp).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            print(f"  FlatParam {i}: max_diff = {diff:.2e}")
    
    # Final verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    print(f"Maximum difference across all GPU counts: {max_diff_overall:.2e}")
    
    threshold = 1e-6
    if max_diff_overall < threshold:
        print(f"✅ ALL GPU COUNTS PRODUCE EXACTLY THE SAME RESULTS!")
        print(f"   (max diff {max_diff_overall:.2e} < threshold {threshold:.2e})")
        return True
    else:
        print(f"✗ DIFFERENCES DETECTED!")
        print(f"   Max diff {max_diff_overall:.2e} exceeds threshold {threshold:.2e}")
        print(f"\n⚠️  BUG IN FSDP IMPLEMENTATION - MUST FIX!")
        
        # Show which comparison failed
        for ws in [2, 4, 8]:
            test_params = results[ws]['params']
            for i, (bp, tp) in enumerate(zip(baseline_params, test_params)):
                diff = (bp - tp).abs().max().item()
                if diff > threshold:
                    print(f"\n  FlatParam {i} @ {ws} GPU: diff = {diff:.2e}")
                    print(f"    First few elements:")
                    print(f"      1 GPU: {bp.flatten()[:5].tolist()}")
                    print(f"      {ws} GPU: {tp.flatten()[:5].tolist()}")
        
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    
    if args.compare:
        success = compare_all()
        sys.exit(0 if success else 1)
    else:
        # Determine world size
        if "WORLD_SIZE" in os.environ:
            ws = os.environ["WORLD_SIZE"]
        else:
            ws = "1"
        
        run_training(ws)

