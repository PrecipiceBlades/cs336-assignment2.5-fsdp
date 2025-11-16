"""Task 4: Backward Pass with All-Gather, Compute, and Reduce-Scatter.

This module implements the backward pass logic for FSDP:
1. All-gather parameters before backward (if they were resharded after forward)
2. Compute backward pass (autograd computes gradients)
3. Reduce-scatter gradients across ranks (sum + scatter)
4. Reshard parameters back to local shard

Key concepts:
- All-gather before backward: Only needed if we resharded after forward
- Reduce-scatter: Combines reduce (sum gradients) and scatter (each rank keeps 1/W)
- post_accumulate_grad_hook: Fires after gradient is accumulated
- Why reduce-scatter not all-reduce: Saves memory (N/W vs N gradient storage)
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple, List
from fsdp.flat_param import FlatParameter
from fsdp.utils import reduce_scatter_tensor, get_world_size, get_rank


def reduce_scatter_grads(
    flat_param: FlatParameter,
    async_op: bool = False
) -> Optional[torch.distributed.Work]:
    """Reduce-scatter gradients: sum across ranks, each rank keeps a shard.
    
    Args:
        flat_param: FlatParameter whose gradients to reduce-scatter
        async_op: If True, return work handle for async operation
    
    Returns:
        Work handle if async_op=True, else None
    
    Example:
        >>> # After backward pass, flat_param.grad contains full gradient
        >>> reduce_scatter_grads(flat_param)
        >>> # Now flat_param.grad contains only local gradient shard
        >>> # Gradient shards from all ranks have been summed
    
    Why reduce-scatter instead of all-reduce?
    - All-reduce: Every rank gets full gradient (N elements per rank)
    - Reduce-scatter: Every rank gets gradient shard (N/W elements per rank)
    - Memory savings: N → N/W per rank
    """
    if async_op:
        raise NotImplementedError("Async reduce-scatter not yet supported")
    
    # Collect gradients from original parameters (views)
    # After backward, gradients are in orig_param.grad, not flat_param.grad
    full_grad_list = []
    
    # DEBUG
    DEBUG = os.environ.get("DEBUG_GRADS") == "1"
    
    if DEBUG:
        print(f"[DEBUG Rank {get_rank()}] Number of orig_params: {len(flat_param._orig_params)}")
    
    for i, orig_param in enumerate(flat_param._orig_params):
        if orig_param.grad is not None:
            grad_flat = orig_param.grad.flatten()
            full_grad_list.append(grad_flat)
            if DEBUG:
                print(f"[DEBUG Rank {get_rank()}] orig_param {i} grad sum: {grad_flat.sum().item():.10f}, numel: {grad_flat.numel()}")
        else:
            # No gradient for this parameter
            zeros = torch.zeros_like(orig_param.data.flatten())
            full_grad_list.append(zeros)
            if DEBUG:
                print(f"[DEBUG Rank {get_rank()}] orig_param {i} grad: None, using zeros")
    
    # Concatenate all gradients into a single flat gradient
    full_grad = torch.cat(full_grad_list)
    
    if DEBUG:
        print(f"[DEBUG Rank {get_rank()}] full_grad sum BEFORE padding: {full_grad.sum().item():.10f}, numel: {full_grad.numel()}")
    
    # For single rank, just copy (no padding needed)
    if flat_param.world_size == 1:
        flat_param.grad = full_grad.clone()
        return None
    
    # Pad gradient to match padded total numel (for reduce-scatter compatibility)
    # CRITICAL: Padding must be zeros so it doesn't affect the reduce-scatter sum
    if full_grad.numel() < flat_param._padded_total_numel:
        padding_size = flat_param._padded_total_numel - full_grad.numel()
        full_grad = torch.cat([
            full_grad,
            torch.zeros(padding_size, dtype=full_grad.dtype, device=full_grad.device)
        ])
    
    if DEBUG and get_rank() == 0:
        print(f"[DEBUG] full_grad sum AFTER padding: {full_grad.sum().item():.10f}")
        print(f"[DEBUG] full_grad numel: {full_grad.numel()}, padded_total: {flat_param._padded_total_numel}")
    
    # Allocate storage for local gradient shard
    local_grad_shard = torch.empty_like(flat_param.data)
    
    # Reduce-scatter: sum gradients and distribute shards
    reduce_scatter_tensor(
        output_tensor=local_grad_shard,
        input_tensor=full_grad,
        async_op=False
    )
    
    if DEBUG and get_rank() == 0:
        print(f"[DEBUG] local_grad_shard sum AFTER reduce-scatter: {local_grad_shard.sum().item():.10f}")
    
    # CRITICAL: Average gradients across world_size (data parallel)
    # reduce-scatter sums gradients from all ranks, but for data parallel
    # we want the average gradient
    # NOTE: Only divide if world_size > 1, for world_size=1 no reduction happens
    if flat_param.world_size > 1:
        local_grad_shard.div_(flat_param.world_size)
    
    if DEBUG and get_rank() == 0:
        print(f"[DEBUG] local_grad_shard sum AFTER averaging: {local_grad_shard.sum().item():.10f}")
    
    # CRITICAL: Zero out any padding in the gradient shard
    # This ensures padding doesn't affect optimizer updates
    # The last rank might have padding in its shard
    shard_start = flat_param._shard_offset
    shard_end = shard_start + flat_param._shard_numel
    
    # If this shard extends into the padding region, zero out the padding part
    if shard_end > flat_param._total_numel:
        # Calculate how much of this shard is valid (not padding)
        valid_size = flat_param._total_numel - shard_start
        if valid_size < flat_param._shard_numel and valid_size >= 0:
            # Zero out the padding portion
            local_grad_shard[valid_size:] = 0.0
    
    # Update flat_param.grad to point to local shard
    flat_param.grad = local_grad_shard
    
    return None


def create_backward_pre_hook(
    flat_param: FlatParameter,
    reshard_after_forward: bool = True
):
    """Create backward pre-hook that all-gathers parameters if needed.
    
    Args:
        flat_param: FlatParameter to manage
        reshard_after_forward: Whether parameters were resharded after forward
    
    Returns:
        Hook function for backward pre-hook
    
    Note:
        We only need to all-gather if we resharded after forward.
        If reshard_after_forward=False, parameters are still full from forward.
    """
    def backward_pre_hook(module: nn.Module, grad_outputs):
        """Hook called before backward pass.
        
        If parameters were resharded after forward, we need to all-gather them
        again before backward can compute gradients w.r.t. full parameters.
        """
        if reshard_after_forward and flat_param._is_sharded:
            # Need to all-gather again for backward
            from fsdp.forward_pass import all_gather_params
            all_gather_params(flat_param)
            flat_param.use_full_param()
    
    return backward_pre_hook


def create_post_accumulate_grad_hook(
    flat_param: FlatParameter,
) -> Callable:
    """Create post-accumulate-grad hook for gradient reduce-scatter.
    
    This hook fires after the gradient has been accumulated in .grad.
    At this point, we can reduce-scatter the gradient.
    
    Args:
        flat_param: FlatParameter to manage
    
    Returns:
        Hook function for post_accumulate_grad_hook
    
    Note:
        post_accumulate_grad_hook is called after backward computes the gradient.
        This is the right place to reduce-scatter gradients.
        
        We register this hook on EACH original parameter, not on flat_param.
        The hook will be called multiple times (once per parameter), but we only
        reduce-scatter once (after all parameters have gradients).
    """
    # Use a flag to ensure we only reduce-scatter once
    hook_state = {'grad_ready_count': 0, 'total_params': len(flat_param._orig_params)}
    
    def post_accumulate_grad_hook(param: nn.Parameter) -> None:
        """Hook called after gradient is accumulated.
        
        At this point:
        - param.grad contains the gradient for this parameter
        - We need to reduce-scatter all gradients after all params are ready
        - We also reshard parameters back to local shard
        """
        hook_state['grad_ready_count'] += 1
        
        # Only reduce-scatter when all parameters have gradients
        if hook_state['grad_ready_count'] == hook_state['total_params']:
            # Reset counter for next backward
            hook_state['grad_ready_count'] = 0
            
            # Reduce-scatter gradients
            reduce_scatter_grads(flat_param)
            
            # Reshard parameters back to local shard
            from fsdp.forward_pass import reshard_params
            reshard_params(flat_param)
    
    return post_accumulate_grad_hook


def register_backward_hooks(
    module: nn.Module,
    flat_param: FlatParameter,
    reshard_after_forward: bool = True
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register backward hooks on a module's parameters.
    
    Args:
        module: Module to register hooks on
        flat_param: FlatParameter for this module
        reshard_after_forward: Whether parameters were resharded after forward
    
    Returns:
        List of hook handles for removal if needed
    
    Example:
        >>> module = nn.Linear(10, 10)
        >>> flat_param = flatten_module_params(module)
        >>> handles = register_backward_hooks(
        ...     module, flat_param, reshard_after_forward=True
        ... )
        >>> # Backward pass will now automatically all-gather, compute, reduce-scatter
        >>> loss.backward()
    """
    handles = []
    
    # Register backward pre-hook
    pre_hook = create_backward_pre_hook(flat_param, reshard_after_forward)
    pre_handle = module.register_full_backward_pre_hook(pre_hook)
    handles.append(pre_handle)
    
    # Register post-accumulate-grad hook on EACH original parameter
    # This is critical: gradients accumulate in orig_params, not flat_param
    post_hook = create_post_accumulate_grad_hook(flat_param)
    for orig_param in flat_param._orig_params:
        if orig_param.requires_grad:
            grad_handle = orig_param.register_post_accumulate_grad_hook(post_hook)
            handles.append(grad_handle)
    
    return handles


# Example usage (for testing):
if __name__ == "__main__":
    print("Test: Backward pass with all-gather and reduce-scatter")
    
    # Create a simple module
    module = nn.Linear(10, 5)
    
    # Create FlatParameter
    from fsdp.flat_param import flatten_module_params
    flat_param = flatten_module_params(module, rank=0, world_size=2)
    
    # Register forward and backward hooks
    from fsdp.forward_pass import register_forward_hooks
    register_forward_hooks(module, flat_param, reshard_after_forward=True)
    register_backward_hooks(module, flat_param, reshard_after_forward=True)
    
    # Test forward and backward
    input_tensor = torch.randn(3, 10, requires_grad=True)
    output = module(input_tensor)
    loss = output.sum()
    loss.backward()
    
    print(f"After backward, flat_param.grad shape: {flat_param.grad.shape if flat_param.grad is not None else None}")
    print(f"Gradient is for local shard: {flat_param.grad.numel() == flat_param.local_shard.numel() if flat_param.grad is not None else 'N/A'}")

