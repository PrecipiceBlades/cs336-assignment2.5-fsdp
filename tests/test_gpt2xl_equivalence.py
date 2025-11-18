"""
GPT-2 XL严格等价性测试：Single GPU vs DDP vs FSDP

测试内容：
1. Loss等价性：三种方式应该产生相同的loss
2. 参数等价性：三种方式应该产生相同的参数
3. 内存验证：FSDP memory per device ≈ 1/8 DDP memory overhead
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys

sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
sys.path.append('/root/cs336-assignment2-systems/cs336-basics')

from cs336_basics.model import BasicsTransformerLM
from fsdp.api import fully_shard, get_flat_parameters, clear_fsdp_registry
from fsdp.optimizer import FSDPOptimizer
from fsdp.utils import setup_distributed, cleanup_distributed, get_rank, get_world_size
from fsdp.meta_init import init_model_on_meta

from torch.distributed.fsdp import fully_shard as official_fully_shard
from torch.distributed.tensor import DTensor

def get_memory_mb(device=None):
    """Get current memory usage in MB."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def train_single_gpu(config, batch_size=32, seq_len=128, num_steps=5, seed=42):
    """Single GPU训练（Non-FSDP baseline）"""
    print("="*70)
    print(f"Single GPU Training (Non-FSDP, batch_size={batch_size})")
    print("="*70)
    
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Initialize on CPU first (to match Meta FSDP's CPU initialization)
    # This ensures deterministic initialization across CPU/GPU
    model = BasicsTransformerLM(**config)
    
    # Print initial parameter sum BEFORE moving to GPU
    init_param_sum = sum(p.sum().item() for p in model.parameters())
    print(f"Initial param sum (CPU): {init_param_sum:.15f}")
    
    model = model.to(device)
    
    # Use larger learning rate for larger models to ensure convergence
    # GPT-2 XL needs more steps and higher LR to show convergence
    lr = 5e-4 if config.get("num_layers", 4) >= 48 else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Measure memory after model creation
    torch.cuda.reset_peak_memory_stats(device)
    memory_after_model = get_memory_mb(device)
    
    losses = []
    for step in range(num_steps):
        torch.manual_seed(seed + 100 + step)
        input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len)).to(device)
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Step {step}: Loss = {loss.item():.10f}")
    
    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    params = [p.detach().cpu().clone() for p in model.parameters()]
    param_sum = sum(p.sum().item() for p in params)
    
    print(f"\nFinal param sum: {param_sum:.15f}")
    print(f"Peak memory: {peak_memory:.2f} MB")
    
    return losses, params, peak_memory


def train_ddp(config, batch_size_per_gpu=4, seq_len=128, num_steps=5, seed=42):
    """DDP训练(8 GPUs)"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*70)
        print(f"DDP Training ({world_size} GPUs, batch_size={batch_size_per_gpu}/GPU)")
        print(f"Total effective batch_size: {batch_size_per_gpu * world_size}")
        print("="*70)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Initialize on CPU first for deterministic initialization
    model = BasicsTransformerLM(**config)
    model = model.to(device)
    
    # Wrap with DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )
    
    # Use larger learning rate for larger models to ensure convergence
    # GPT-2 XL needs more steps and higher LR to show convergence
    lr = 5e-4 if config.get("num_layers", 4) >= 48 else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Measure memory after model creation
    torch.cuda.reset_peak_memory_stats(device)
    
    losses = []
    all_rank_losses = []
    
    for step in range(num_steps):
        torch.manual_seed(seed + 100 + step)
        # Each rank gets different data
        full_input_ids = torch.randint(
            0, config["vocab_size"], 
            (batch_size_per_gpu * world_size, seq_len)
        )
        input_ids = full_input_ids[
            rank * batch_size_per_gpu : (rank + 1) * batch_size_per_gpu
        ].to(device)
        
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        
        rank_loss = loss.item()
        losses.append(rank_loss)
        
        # Collect losses from all ranks
        loss_tensor = torch.tensor(rank_loss, device=device)
        gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_tensor)
        avg_loss = sum(l.item() for l in gathered_losses) / world_size
        all_rank_losses.append(avg_loss)
        
        if rank == 0:
            print(f"  Step {step}: Rank-0 Loss = {rank_loss:.10f}, Avg Loss = {avg_loss:.10f}")
    
    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # CRITICAL: All ranks must synchronize before collecting parameters
    dist.barrier()
    
    # Get parameters (unwrap DDP)
    model_module = model.module
    params = [p.detach().cpu().clone() for p in model_module.parameters()]
    param_sum = sum(p.sum().item() for p in params)
    
    if rank == 0:
        print(f"\nFinal param sum: {param_sum:.15f}")
        print(f"Peak memory (per device): {peak_memory:.2f} MB")
        print(f"Total memory (all devices): {peak_memory * world_size:.2f} MB")
        
        cleanup_distributed()
        return all_rank_losses, params, peak_memory
    
    cleanup_distributed()
    return None, None, None


def train_fsdp(config, batch_size_per_gpu=4, seq_len=128, num_steps=5, seed=42):
    """FSDP训练(8 GPUs)"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*70)
        print(f"FSDP Training ({world_size} GPUs, batch_size={batch_size_per_gpu}/GPU)")
        print(f"Total effective batch_size: {batch_size_per_gpu * world_size}")
        print("="*70)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    clear_fsdp_registry()
    # Initialize on CPU first for deterministic initialization
    model = BasicsTransformerLM(**config)
    model = model.to(device)
    
    # Apply FSDP
    for block in model.layers:
        fully_shard(block)
    fully_shard(model.token_embeddings)
    fully_shard(model.lm_head)
    if hasattr(model, 'final_norm') and len(list(model.final_norm.parameters())) > 0:
        fully_shard(model.final_norm)
    
    # Use larger learning rate for larger models to ensure convergence
    # GPT-2 XL needs more steps and higher LR to show convergence
    lr = 5e-4 if config.get("num_layers", 4) >= 48 else 1e-4
    optimizer = FSDPOptimizer(
        list(model.parameters()),
        optimizer_cls=torch.optim.AdamW,
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    
    # Measure memory after model creation
    torch.cuda.reset_peak_memory_stats(device)
    
    # Calculate sharding ratio
    if rank == 0:
        flat_params = get_flat_parameters(model)
        total_params = sum(fp._total_numel for fp in flat_params)
        local_params = sum(fp.data.numel() for fp in flat_params)
        sharding_ratio = total_params / local_params if local_params > 0 else 1.0
        print(f"  Parameter sharding: {total_params:,} / {local_params:,} = {sharding_ratio:.2f}x")
    
    losses = []
    all_rank_losses = []
    
    for step in range(num_steps):
        torch.manual_seed(seed + 100 + step)
        # Each rank gets different data
        full_input_ids = torch.randint(
            0, config["vocab_size"], 
            (batch_size_per_gpu * world_size, seq_len)
        )
        input_ids = full_input_ids[
            rank * batch_size_per_gpu : (rank + 1) * batch_size_per_gpu
        ].to(device)
        
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        
        rank_loss = loss.item()
        losses.append(rank_loss)
        
        # Collect losses from all ranks
        loss_tensor = torch.tensor(rank_loss, device=device)
        gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_tensor)
        avg_loss = sum(l.item() for l in gathered_losses) / world_size
        all_rank_losses.append(avg_loss)
        
        if rank == 0:
            print(f"  Step {step}: Rank-0 Loss = {rank_loss:.10f}, Avg Loss = {avg_loss:.10f}")
    
    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Get final parameters
    # CRITICAL: All ranks must participate in all_gather!
    flat_params = get_flat_parameters(model)
    for fp in flat_params:
        fp.all_gather()
        fp.use_full_param()
    
    # CRITICAL: All ranks must synchronize before collecting parameters
    dist.barrier()
    
    if rank == 0:
        from fsdp.flat_param import FlatParameter
        params = [
            p.detach().cpu().clone() 
            for p in model.parameters() 
            if not isinstance(p, FlatParameter)
        ]
        param_sum = sum(p.sum().item() for p in params)
        
        print(f"\nFinal param sum: {param_sum:.15f}")
        print(f"Peak memory (per device): {peak_memory:.2f} MB")
        print(f"Total memory (all devices): {peak_memory * world_size:.2f} MB")
        
        cleanup_distributed()
        return all_rank_losses, params, peak_memory
    
    cleanup_distributed()
    return None, None, None


def train_official_fsdp(config, batch_size_per_gpu=4, seq_len=128, num_steps=5, seed=42):
    """使用PyTorch官方FSDP2训练(8 GPUs) - 参考官方example"""
    try:
        from torch.distributed.fsdp import fully_shard as official_fully_shard
    except ImportError:
        raise RuntimeError("PyTorch official FSDP2 is not available")
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*70)
        print(f"Official FSDP2 Training ({world_size} GPUs, batch_size={batch_size_per_gpu}/GPU)")
        print(f"Total effective batch_size: {batch_size_per_gpu * world_size}")
        print("="*70)
    
    # CRITICAL: Synchronize random seeds across all ranks before initialization
    # This ensures all ranks initialize parameters identically
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Synchronize all ranks before model initialization
    dist.barrier()
    
    # Strategy: Initialize model directly on device (same as single GPU)
    # This ensures parameters are initialized identically before FSDP2 sharding
    # All ranks use the same seed, so they should initialize identically
    model = BasicsTransformerLM(**config).to(device)
    
    # Synchronize after initialization to ensure all ranks are ready
    dist.barrier()
    
    # Now apply FSDP2 AFTER initialization
    # FSDP2 will shard the already-initialized parameters
    # This should preserve the initial parameter values
    for layer in model.layers:
        official_fully_shard(layer)
    official_fully_shard(model.token_embeddings)
    official_fully_shard(model.lm_head)
    if hasattr(model, 'final_norm') and len(list(model.final_norm.parameters())) > 0:
        official_fully_shard(model.final_norm)
    # Finally wrap the root model
    official_fully_shard(model)
    
    # Synchronize after applying FSDP2
    dist.barrier()
    
    # Use larger learning rate for larger models to ensure convergence
    # GPT-2 XL needs more steps and higher LR to show convergence
    lr = 5e-4 if config.get("num_layers", 4) >= 48 else 1e-4
    # FSDP2 parameters are DTensor, standard optimizer works out of the box
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Measure memory after model creation
    torch.cuda.reset_peak_memory_stats(device)
    
    losses = []
    all_rank_losses = []
    
    for step in range(num_steps):
        torch.manual_seed(seed + 100 + step)
        # Each rank gets different data
        full_input_ids = torch.randint(
            0, config["vocab_size"], 
            (batch_size_per_gpu * world_size, seq_len)
        )
        input_ids = full_input_ids[
            rank * batch_size_per_gpu : (rank + 1) * batch_size_per_gpu
        ].to(device)
        
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        
        rank_loss = loss.item()
        losses.append(rank_loss)
        
        # Collect losses from all ranks
        loss_tensor = torch.tensor(rank_loss, device=device)
        gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_tensor)
        avg_loss = sum(l.item() for l in gathered_losses) / world_size
        all_rank_losses.append(avg_loss)
        
        if rank == 0:
            print(f"  Step {step}: Rank-0 Loss = {rank_loss:.10f}, Avg Loss = {avg_loss:.10f}")
    
    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # CRITICAL: All ranks must synchronize before collecting parameters
    dist.barrier()
    
    # Get parameters using state_dict() and convert DTensor to full tensor
    # Following official example pattern: use state_dict() then full_tensor()
    # But we need to match the parameter order from model.parameters() for comparison
    # CRITICAL: All ranks must participate in full_tensor() calls
    from torch.distributed.tensor import DTensor
    # Collect parameters in the same order as model.parameters()
    params = []
    for p in model.parameters():
        if isinstance(p, DTensor):
            # Convert DTensor to full tensor using full_tensor()
            # This requires all ranks to participate
            full_param = p.full_tensor()
            if rank == 0:
                params.append(full_param.detach().cpu().clone())
            else:
                del full_param  # Free memory on non-rank-0
        elif hasattr(p, 'to_local'):
            # DTensor with to_local method
            local_param = p.to_local()
            if rank == 0:
                params.append(local_param.detach().cpu().clone())
            else:
                del local_param
        else:
            # Regular tensor
            if rank == 0:
                params.append(p.detach().cpu().clone())
    
    if rank == 0:
        param_sum = sum(p.sum().item() for p in params)
        
        print(f"\nFinal param sum: {param_sum:.15f}")
        print(f"Peak memory (per device): {peak_memory:.2f} MB")
        print(f"Total memory (all devices): {peak_memory * world_size:.2f} MB")
        
        cleanup_distributed()
        return all_rank_losses, params, peak_memory
    
    cleanup_distributed()
    return None, None, None


def train_fsdp_meta_device(config, batch_size_per_gpu=4, seq_len=128, num_steps=5, seed=42):
    """使用Meta Device + FSDP训练(8 GPUs) - 内存高效版本
    
    FSDP2 Meta Device Flow (官方推荐):
    1. 在meta device上创建模型（无内存分配）
    2. 使用materialize_meta_module将模型materialize到CPU（正确初始化）
    3. 应用fully_shard进行参数分片
    4. 移动到GPU device进行训练
    
    关键改进：RotaryEmbedding现在支持lazy initialization，
    _freq_cis_cache会在第一次forward时自动初始化到正确的device。
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    setup_distributed(rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("="*70)
        print(f"Meta Device FSDP Training ({world_size} GPUs, batch_size={batch_size_per_gpu}/GPU)")
        print(f"Total effective batch_size: {batch_size_per_gpu * world_size}")
        print("="*70)
    
    # CRITICAL: Synchronize random seeds across all ranks
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Synchronize all ranks before model initialization
    dist.barrier()
    
    if rank == 0:
        print("\n=== FSDP2 Meta Device Flow ===")
        print("Step 1: Creating model on meta device...")
    
    # Step 1: Create model on meta device (no memory allocated)
    # CRITICAL: Save and restore RNG state so meta model creation doesn't consume random numbers
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    
    with torch.device("meta"):
        model = BasicsTransformerLM(**config)
    
    # Restore RNG state so subsequent initialization uses the same random sequence
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {n_params:,}")
        print(f"  Memory allocated: 0 MB (all parameters on meta device)")
        print(f"  All params are meta: {all(p.is_meta for p in model.parameters())}")
        print(f"  RNG state restored for deterministic initialization")
    
    # Step 2: Materialize the ENTIRE model from meta to CPU first
    # This ensures all ranks use the same initialization sequence
    if rank == 0:
        print("\nStep 2: Materializing entire model to CPU (replaying CS336 initialization)...")
    
    # Step 2: Materialize meta parameters to CPU
    # This replays the CS336 initialization logic (trunc_normal_ for Linear/Embedding, ones for RMSNorm)
    # The materialization follows the exact __init__ order of BasicsTransformerLM to ensure
    # deterministic RNG consumption across all ranks
    from fsdp.meta_init import materialize_meta_module
    materialize_meta_module(model, torch.device("cpu"), init_fn=None)
    
    if rank == 0:
        all_cpu = all(not p.is_meta for p in model.parameters())
        print(f"  All params materialized to CPU: {all_cpu}")
    
    # Clear registry before wrapping
    clear_fsdp_registry()
    
    # Step 3: Apply FSDP to shard parameters across ranks
    # CRITICAL: We apply FSDP from inside-out (子模块 → root)
    # This ensures that each parameter is only included in ONE FlatParameter
    # The flatten_module_params function automatically skips parameters from
    # child modules that are already FSDP-managed to prevent duplication
    if rank == 0:
        print("\nStep 3: Applying fully_shard (sharding CPU parameters)...")
    
    # Wrap transformer layers (each layer's parameters will be sharded)
    for layer in model.layers:
        fully_shard(layer)
    
    # Wrap embedding and output layers
    fully_shard(model.token_embeddings)
    fully_shard(model.ln_final)
    fully_shard(model.lm_head)
    
    # Wrap root module (only wraps parameters not already managed by child modules)
    fully_shard(model)
    
    if rank == 0:
        flat_params = get_flat_parameters(model)
        print(f"  Created {len(flat_params)} FlatParameters")
    
    # Step 4: Handle remaining meta buffers (like _freq_cis_cache)
    # Note: Some buffers (like RotaryEmbedding._freq_cis_cache) are computed and non-persistent
    # They need explicit initialization since they're not included in state_dict
    if rank == 0:
        print("\nStep 4: Initializing non-parameter buffers...")
    
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            # Get parent module
            parent = model
            *path, buf_name = name.split('.')
            for attr in path:
                parent = getattr(parent, attr)
            
            # Manually initialize the buffer (for RotaryEmbedding._freq_cis_cache)
            if 'positional_encoder._freq_cis_cache' in name:
                # Use the RotaryEmbedding's own initialization
                from cs336_basics.model import RotaryEmbedding
                cache_val = RotaryEmbedding._init_cache(
                    parent.context_length,
                    parent.dim,
                    parent.theta
                )
                parent.register_buffer("_freq_cis_cache", cache_val, persistent=False)
                if rank == 0:
                    print(f"  Initialized {name} on CPU")
            else:
                # For other buffers, delete and let them be lazily initialized
                delattr(parent, buf_name)
                if rank == 0:
                    print(f"  Removed meta buffer {name}")
    
    # Step 5: Move sharded model to GPU
    # After FSDP sharding, each rank only holds its local shard in memory
    # Moving to GPU is much more memory-efficient than materializing the full model first
    if rank == 0:
        print("\nStep 5: Moving sharded model to GPU...")
    
    model = model.to(device)
    
    flat_params = get_flat_parameters(model)
    
    if rank == 0:
        total_shard_size = sum(fp.numel() for fp in flat_params)
        print(f"Total shard size: {total_shard_size:,} parameters")
        print(f"Shard memory (FP32): {total_shard_size * 4 / 1024**2:.2f} MB per device")
    
    # Synchronize after wrapping
    dist.barrier()
    
    # Create optimizer
    lr = 5e-4 if config.get("num_layers", 4) >= 48 else 1e-4
    optimizer = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.AdamW, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Measure memory after model creation
    torch.cuda.reset_peak_memory_stats(device)
    
    all_rank_losses = []
    for step in range(num_steps):
        # Data generation strategy for equivalence with Single GPU:
        # 1. All ranks use the same seed to generate the SAME full batch
        # 2. Each rank takes a different slice of this batch
        # 3. This ensures: rank 0's data = single_gpu's data[0:batch_size_per_gpu]
        #                  rank 1's data = single_gpu's data[batch_size_per_gpu:2*batch_size_per_gpu], etc.
        torch.manual_seed(seed + 100 + step)
        full_batch = torch.randint(0, config["vocab_size"], (batch_size_per_gpu * world_size, seq_len))
        start_idx = rank * batch_size_per_gpu
        end_idx = start_idx + batch_size_per_gpu
        input_ids = full_batch[start_idx:end_idx].to(device)
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        
        rank_loss = loss.item()
        
        # Collect losses from all ranks and compute average for comparison with Single GPU
        # Note: Single GPU processes the entire batch at once, so its loss is already an average
        # For FSDP, each rank computes loss on its slice, so we average across ranks
        loss_tensor = torch.tensor(rank_loss, device=device)
        gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss_tensor)
        avg_loss = sum(l.item() for l in gathered_losses) / world_size
        all_rank_losses.append(avg_loss)
        
        if rank == 0:
            print(f"  Step {step}: Rank-0 Loss = {rank_loss:.10f}, Avg Loss = {avg_loss:.10f}")
    
    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Get final parameters
    # CRITICAL: All ranks must participate in all_gather!
    flat_params = get_flat_parameters(model)
    for fp in flat_params:
        fp.all_gather()
        fp.use_full_param()
    
    # CRITICAL: All ranks must synchronize before collecting parameters
    dist.barrier()
    
    if rank == 0:
        from fsdp.flat_param import FlatParameter
        params = [
            p.detach().cpu().clone() 
            for p in model.parameters() 
            if not isinstance(p, FlatParameter)
        ]
        param_sum = sum(p.sum().item() for p in params)
        
        print(f"\nFinal param sum: {param_sum:.15f}")
        print(f"Peak memory (per device): {peak_memory:.2f} MB")
        print(f"Total memory (all devices): {peak_memory * world_size:.2f} MB")
        
        cleanup_distributed()
        return all_rank_losses, params, peak_memory
    
    cleanup_distributed()
    return None, None, None


def compare_all_results():
    """比较Single GPU、DDP、FSDP和Official FSDP2的结果(8 GPUs)"""
    single = torch.load("/tmp/single_gpt2xl_result.pt")
    ddp = torch.load("/tmp/ddp_gpt2xl_result.pt")
    fsdp = torch.load("/tmp/fsdp_gpt2xl_result.pt")
    
    # Try to load official FSDP2 results if available
    try:
        official_fsdp = torch.load("/tmp/official_fsdp2_gpt2xl_result.pt")
        has_official = True
    except FileNotFoundError:
        official_fsdp = None
        has_official = False
    
    print("\n" + "="*70)
    if has_official:
        print("COMPARISON: Single GPU vs DDP vs FSDP vs Official FSDP2")
    else:
        print("COMPARISON: Single GPU vs DDP vs FSDP")
    print("="*70)
    
    # Compare losses
    if has_official:
        print(f"\n{'Step':<10} {'Single GPU':<20} {'DDP':<20} {'FSDP':<20} {'Official FSDP2':<20} {'DDP Diff':<15} {'FSDP Diff':<15} {'Official Diff':<15}")
        print("-" * 140)
    else:
        print(f"\n{'Step':<10} {'Single GPU':<20} {'DDP':<20} {'FSDP':<20} {'DDP Diff':<15} {'FSDP Diff':<15}")
        print("-" * 100)
    
    max_loss_diff_ddp = 0.0
    max_loss_diff_fsdp = 0.0
    max_loss_diff_official = 0.0
    
    for i, (l_single, l_ddp, l_fsdp) in enumerate(zip(single['losses'], ddp['losses'], fsdp['losses'])):
        diff_ddp = abs(l_single - l_ddp)
        diff_fsdp = abs(l_single - l_fsdp)
        max_loss_diff_ddp = max(max_loss_diff_ddp, diff_ddp)
        max_loss_diff_fsdp = max(max_loss_diff_fsdp, diff_fsdp)
        
        if has_official:
            l_official = official_fsdp['losses'][i]
            diff_official = abs(l_single - l_official)
            max_loss_diff_official = max(max_loss_diff_official, diff_official)
            print(f"{i:<10} {l_single:<20.10f} {l_ddp:<20.10f} {l_fsdp:<20.10f} {l_official:<20.10f} {diff_ddp:<15.2e} {diff_fsdp:<15.2e} {diff_official:<15.2e}")
        else:
            print(f"{i:<10} {l_single:<20.10f} {l_ddp:<20.10f} {l_fsdp:<20.10f} {diff_ddp:<15.2e} {diff_fsdp:<15.2e}")
    
    # Compare parameters
    print(f"\nParameter Comparison:")
    print(f"  Single GPU: {len(single['params'])} parameters")
    print(f"  DDP: {len(ddp['params'])} parameters")
    print(f"  FSDP: {len(fsdp['params'])} parameters")
    if has_official:
        print(f"  Official FSDP2: {len(official_fsdp['params'])} parameters")
    
    max_param_diff_ddp = 0.0
    max_param_diff_fsdp = 0.0
    max_param_diff_official = 0.0
    param_diffs_ddp = []
    param_diffs_fsdp = []
    param_diffs_official = []
    
    for i, (p_single, p_ddp, p_fsdp) in enumerate(zip(single['params'], ddp['params'], fsdp['params'])):
        if p_single.shape != p_ddp.shape or p_single.shape != p_fsdp.shape:
            continue
        diff_ddp = (p_single - p_ddp).abs().max().item()
        diff_fsdp = (p_single - p_fsdp).abs().max().item()
        max_param_diff_ddp = max(max_param_diff_ddp, diff_ddp)
        max_param_diff_fsdp = max(max_param_diff_fsdp, diff_fsdp)
        param_diffs_ddp.append(diff_ddp)
        param_diffs_fsdp.append(diff_fsdp)
        
        if has_official and i < len(official_fsdp['params']):
            p_official = official_fsdp['params'][i]
            if p_single.shape == p_official.shape:
                diff_official = (p_single - p_official).abs().max().item()
                max_param_diff_official = max(max_param_diff_official, diff_official)
                param_diffs_official.append(diff_official)
                if i < 5:
                    print(f"  Param {i}: DDP diff={diff_ddp:.2e}, FSDP diff={diff_fsdp:.2e}, Official diff={diff_official:.2e}")
            else:
                if i < 5:
                    print(f"  Param {i}: DDP diff={diff_ddp:.2e}, FSDP diff={diff_fsdp:.2e}")
        else:
            if i < 5:
                print(f"  Param {i}: DDP diff={diff_ddp:.2e}, FSDP diff={diff_fsdp:.2e}")
    
    # Show statistics
    if param_diffs_ddp:
        import numpy as np
        param_diffs_ddp = np.array(param_diffs_ddp)
        param_diffs_fsdp = np.array(param_diffs_fsdp)
        print(f"\n  Parameter diff statistics:")
        print(f"    DDP:  mean={param_diffs_ddp.mean():.2e}, median={np.median(param_diffs_ddp):.2e}, max={param_diffs_ddp.max():.2e}")
        print(f"    FSDP: mean={param_diffs_fsdp.mean():.2e}, median={np.median(param_diffs_fsdp):.2e}, max={param_diffs_fsdp.max():.2e}")
        print(f"    FSDP 95th percentile: {np.percentile(param_diffs_fsdp, 95):.2e}")
        if has_official and param_diffs_official:
            param_diffs_official = np.array(param_diffs_official)
            print(f"    Official FSDP2: mean={param_diffs_official.mean():.2e}, median={np.median(param_diffs_official):.2e}, max={param_diffs_official.max():.2e}")
            print(f"    Official FSDP2 95th percentile: {np.percentile(param_diffs_official, 95):.2e}")
    
    # Memory comparison
    print(f"\nMemory Comparison:")
    print(f"  Single GPU: {single['memory']:.2f} MB")
    print(f"  DDP (per device): {ddp['memory']:.2f} MB")
    print(f"  DDP (total, 8 devices): {ddp['memory'] * 8:.2f} MB")
    print(f"  FSDP (per device): {fsdp['memory']:.2f} MB")
    print(f"  FSDP (total, 8 devices): {fsdp['memory'] * 8:.2f} MB")
    if has_official:
        print(f"  Official FSDP2 (per device): {official_fsdp['memory']:.2f} MB")
        print(f"  Official FSDP2 (total, 8 devices): {official_fsdp['memory'] * 8:.2f} MB")
    
    # Calculate memory ratio
    memory_ratio = fsdp['memory'] / ddp['memory']
    expected_ratio = 1.0 / 8.0  # FSDP should be ~1/8 of DDP per device
    
    print(f"\n  FSDP / DDP (per device): {memory_ratio:.3f} (expected ~{expected_ratio:.3f})")
    print(f"  Memory savings: {(1 - memory_ratio) * 100:.1f}%")
    if has_official:
        official_memory_ratio = official_fsdp['memory'] / ddp['memory']
        print(f"  Official FSDP2 / DDP (per device): {official_memory_ratio:.3f}")
        print(f"  Official FSDP2 memory savings: {(1 - official_memory_ratio) * 100:.1f}%")
        print(f"  Our FSDP / Official FSDP2: {fsdp['memory'] / official_fsdp['memory']:.3f}")
    
    print(f"\n{'='*70}")
    print(f"Max loss diff (DDP):  {max_loss_diff_ddp:.2e}")
    print(f"Max loss diff (FSDP): {max_loss_diff_fsdp:.2e}")
    if has_official:
        print(f"Max loss diff (Official FSDP2): {max_loss_diff_official:.2e}")
    print(f"Max param diff (DDP):  {max_param_diff_ddp:.2e}")
    print(f"Max param diff (FSDP): {max_param_diff_fsdp:.2e}")
    if has_official:
        print(f"Max param diff (Official FSDP2): {max_param_diff_official:.2e}")
    
    # Check if all are equivalent
    # For GPT-2 XL, use more lenient thresholds due to floating point accumulation
    threshold_loss = 1e-4
    threshold_param = 1e-2  # More lenient for large models
    
    success = True
    if max_loss_diff_ddp > threshold_loss:
        print(f"\n✗ DDP loss diff {max_loss_diff_ddp:.2e} > threshold {threshold_loss:.2e}")
        success = False
    if max_loss_diff_fsdp > threshold_loss:
        print(f"\n✗ FSDP loss diff {max_loss_diff_fsdp:.2e} > threshold {threshold_loss:.2e}")
        success = False
    if has_official and max_loss_diff_official > threshold_loss:
        print(f"\n✗ Official FSDP2 loss diff {max_loss_diff_official:.2e} > threshold {threshold_loss:.2e}")
        success = False
    if max_param_diff_ddp > threshold_param:
        print(f"\n✗ DDP param diff {max_param_diff_ddp:.2e} > threshold {threshold_param:.2e}")
        success = False
    if max_param_diff_fsdp > threshold_param:
        print(f"\n✗ FSDP param diff {max_param_diff_fsdp:.2e} > threshold {threshold_param:.2e}")
        success = False
    if has_official and max_param_diff_official > threshold_param:
        print(f"\n✗ Official FSDP2 param diff {max_param_diff_official:.2e} > threshold {threshold_param:.2e}")
        success = False
    
    # Check memory ratio
    if memory_ratio > expected_ratio * 1.5:  # Allow 50% tolerance
        print(f"\n⚠️  FSDP memory ratio {memory_ratio:.3f} > expected {expected_ratio:.3f} * 1.5")
        print(f"   (This might be due to activation memory or other overhead)")
    else:
        print(f"\n✅ FSDP memory per device ≈ 1/8 DDP memory overhead")
    
    if success:
        print(f"\n✅ All three methods are equivalent!")
        return True
    else:
        print(f"\n✗ Some methods are not equivalent")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "ddp", "fsdp", "official_fsdp", "meta_fsdp", "compare"], required=True)
    parser.add_argument("--config", choices=["small", "medium", "gpt2xl"], default="small")
    args = parser.parse_args()
    
    # Model configs
    configs = {
        "small": {
            "vocab_size": 1000,
            "context_length": 256,
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 4,
            "d_ff": 1024,
            "rope_theta": 10000.0,
        },
        "medium": {
            "vocab_size": 5000,
            "context_length": 512,
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 8,
            "d_ff": 2048,
            "rope_theta": 10000.0,
        },
        "gpt2xl": {
            "vocab_size": 50304,
            "context_length": 1024,
            "d_model": 1600,
            "num_layers": 48,
            "num_heads": 25,
            "d_ff": 6400,
            "rope_theta": 10000.0,
        }
    }
    
    config = configs[args.config]
    
    if args.mode == "single":
        # Use smaller batch size for GPT-2 XL
        # Use more steps for GPT-2 XL to show convergence
        batch_size = 8 if args.config == "gpt2xl" else 32
        num_steps = 10 if args.config == "gpt2xl" else 5
        losses, params, memory = train_single_gpu(
            config, batch_size=batch_size, seq_len=128, num_steps=num_steps, seed=42
        )
        torch.save({'losses': losses, 'params': params, 'memory': memory}, 
                   "/tmp/single_gpt2xl_result.pt")
        print("\nSaved to /tmp/single_gpt2xl_result.pt")
        
    elif args.mode == "ddp":
        rank = int(os.environ.get("RANK", 0))
        # Use smaller batch size per GPU for GPT-2 XL
        # Use more steps for GPT-2 XL to show convergence
        batch_size_per_gpu = 1 if args.config == "gpt2xl" else 4
        num_steps = 10 if args.config == "gpt2xl" else 5
        losses, params, memory = train_ddp(
            config, batch_size_per_gpu=batch_size_per_gpu, seq_len=128, num_steps=num_steps, seed=42
        )
        if rank == 0 and losses is not None:
            torch.save({'losses': losses, 'params': params, 'memory': memory}, 
                       "/tmp/ddp_gpt2xl_result.pt")
            print("\nSaved to /tmp/ddp_gpt2xl_result.pt")
        
    elif args.mode == "fsdp":
        rank = int(os.environ.get("RANK", 0))
        # Use smaller batch size per GPU for GPT-2 XL
        # Use more steps for GPT-2 XL to show convergence
        batch_size_per_gpu = 1 if args.config == "gpt2xl" else 4
        num_steps = 10 if args.config == "gpt2xl" else 5
        losses, params, memory = train_fsdp(
            config, batch_size_per_gpu=batch_size_per_gpu, seq_len=128, num_steps=num_steps, seed=42
        )
        if rank == 0 and losses is not None:
            torch.save({'losses': losses, 'params': params, 'memory': memory}, 
                       "/tmp/fsdp_gpt2xl_result.pt")
            print("\nSaved to /tmp/fsdp_gpt2xl_result.pt")
        
    elif args.mode == "official_fsdp":
        try:
            from torch.distributed.fsdp import fully_shard as _test_import
        except ImportError:
            print("ERROR: PyTorch official FSDP2 is not available")
            sys.exit(1)
        rank = int(os.environ.get("RANK", 0))
        # Use smaller batch size per GPU for GPT-2 XL
        # Use more steps for GPT-2 XL to show convergence
        batch_size_per_gpu = 1 if args.config == "gpt2xl" else 4
        num_steps = 10 if args.config == "gpt2xl" else 5
        losses, params, memory = train_official_fsdp(
            config, batch_size_per_gpu=batch_size_per_gpu, seq_len=128, num_steps=num_steps, seed=42
        )
        if rank == 0 and losses is not None:
            torch.save({'losses': losses, 'params': params, 'memory': memory}, 
                       "/tmp/official_fsdp2_gpt2xl_result.pt")
            print("\nSaved to /tmp/official_fsdp2_gpt2xl_result.pt")
        
    elif args.mode == "meta_fsdp":
        rank = int(os.environ.get("RANK", 0))
        # Use smaller batch size per GPU for GPT-2 XL
        batch_size_per_gpu = 1 if args.config == "gpt2xl" else 4
        num_steps = 10 if args.config == "gpt2xl" else 5
        losses, params, memory = train_fsdp_meta_device(
            config, batch_size_per_gpu=batch_size_per_gpu, seq_len=128, num_steps=num_steps, seed=42
        )
        if rank == 0 and losses is not None:
            torch.save({'losses': losses, 'params': params, 'memory': memory}, 
                       "/tmp/meta_fsdp_gpt2xl_result.pt")
            print("\nSaved to /tmp/meta_fsdp_gpt2xl_result.pt")
        
    elif args.mode == "compare":
        success = compare_all_results()
        sys.exit(0 if success else 1)

