"""FSDP2-style API: from fsdp import fully_shard

This module provides a FSDP2-inspired API that wraps modules in-place.
Unlike FSDP1 which returns a wrapped module, fully_shard() modifies the module
in-place and returns the same module (now with FSDP functionality).

Key differences from our previous implementation:
1. fully_shard(module) - modifies module in-place
2. Automatic hook registration
3. More Pythonic API
4. Support for meta device initialization

Example usage:
    # Standard usage (model on device)
    model = Transformer().cuda()
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    
    # Meta device usage (memory efficient for large models)
    with torch.device("meta"):
        model = Transformer()
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable
from fsdp.flat_param import flatten_module_params, FlatParameter
from fsdp.forward_pass import register_forward_hooks
from fsdp.backward_pass import register_backward_hooks
from fsdp.meta_init import materialize_meta_module


# Global registry to track FSDP-wrapped modules and their FlatParameters
_FSDP_MODULE_REGISTRY = {}


def fully_shard(
    module: nn.Module,
    *,
    reshard_after_forward: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
) -> nn.Module:
    """Apply FSDP to a module (FSDP2-style API).
    
    This function modifies the module in-place by:
    1. Materializing meta tensors if module is on meta device
    2. Flattening and sharding its parameters into a FlatParameter
    3. Registering forward/backward hooks for all-gather and reduce-scatter
    4. Storing the FlatParameter in a global registry
    
    Args:
        module: Module to apply FSDP to (can be on meta device)
        reshard_after_forward: Whether to reshard parameters after forward pass.
            True: saves memory but requires re-gather in backward
            False: keeps parameters for backward, saves communication
        rank: Current rank (default: 0 for single-GPU)
        world_size: World size (default: 1 for single-GPU)
        param_init_fn: Optional function to initialize parameters after materialization.
            If module is on meta device and this is None, uses default initialization.
            Signature: (module: nn.Module) -> None
    
    Returns:
        The same module (modified in-place)
    
    Example:
        Standard usage:
        >>> from fsdp import fully_shard
        >>> model = Transformer().cuda()
        >>> for layer in model.layers:
        ...     fully_shard(layer)
        >>> fully_shard(model)
        
        Meta device usage (memory efficient):
        >>> with torch.device("meta"):
        ...     model = Transformer()
        >>> for layer in model.layers:
        ...     fully_shard(layer)  # Materializes only this layer's shard
        >>> fully_shard(model)  # Materializes only root module's shard
    
    Comparison with FSDP1:
        FSDP1: model = FSDP(model, auto_wrap_policy=...)
        FSDP2: fully_shard(model); for layer in model.layers: fully_shard(layer)
    """
    if rank is None:
        from fsdp.utils import get_rank
        rank = get_rank()
    
    if world_size is None:
        from fsdp.utils import get_world_size
        world_size = get_world_size()
    
    # Skip if module already has FSDP applied
    if id(module) in _FSDP_MODULE_REGISTRY:
        return module
    
    # Check if module has parameters (check recursively for container modules)
    params = list(module.parameters(recurse=True))
    if len(params) == 0:
        # No parameters to shard, but still register as FSDP-wrapped
        _FSDP_MODULE_REGISTRY[id(module)] = None
        return module
    
    # Edge case: Check if all parameters are already managed by FSDP child modules
    # This happens when we call fully_shard on a container module after wrapping all its children
    # Example: fully_shard(model) after fully_shard(layer) for all layers
    from fsdp.flat_param import _is_fsdp_managed_recursively
    all_children_managed = all(
        _is_fsdp_managed_recursively(child) or len(list(child.parameters(recurse=True))) == 0
        for child in module.children()
    )
    has_direct_params = len(list(module.parameters(recurse=False))) > 0
    
    if all_children_managed and not has_direct_params:
        # All parameters are already sharded in child modules
        # No need to create another FlatParameter for this module
        # Just mark it as FSDP-managed for consistency
        _FSDP_MODULE_REGISTRY[id(module)] = None
        module._is_fsdp_managed = True
        return module
    
    # Check if module is on meta device
    has_meta = any(p.is_meta for p in params)
    
    if has_meta:
        # Materialize meta tensors to CPU first (will be sharded immediately)
        # This is memory efficient because we only materialize temporarily
        materialize_meta_module(module, torch.device("cpu"), param_init_fn)
    
    # Flatten and shard parameters
    flat_param = flatten_module_params(
        module,
        rank=rank,
        world_size=world_size
    )
    
    # Register hooks
    register_forward_hooks(
        module,
        flat_param,
        reshard_after_forward=reshard_after_forward
    )
    
    register_backward_hooks(
        module,
        flat_param,
        reshard_after_forward=reshard_after_forward
    )
    
    # Store in registry
    _FSDP_MODULE_REGISTRY[id(module)] = flat_param
    
    # Add FSDP marker attribute
    module._is_fsdp_managed = True
    module._fsdp_flat_param = flat_param
    
    return module


def get_flat_parameters(model: nn.Module) -> List[FlatParameter]:
    """Get all FlatParameters from a FSDP-wrapped model.
    
    Args:
        model: FSDP-wrapped model
    
    Returns:
        List of FlatParameters (one per FSDP-wrapped module)
    
    Example:
        >>> flat_params = get_flat_parameters(model)
        >>> optimizer = FSDPOptimizer(flat_params, optimizer_cls=torch.optim.Adam, lr=1e-3)
    
    Note:
        This function only collects FlatParameters from direct FSDP-wrapped modules,
        not from their children. For example, if you wrap model.layers[0] and model,
        this will return 2 FlatParameters (one for layers[0], one for model's direct parameters).
    """
    flat_params = []
    seen_ids = set()
    
    for module in model.modules():
        module_id = id(module)
        if module_id in _FSDP_MODULE_REGISTRY and module_id not in seen_ids:
            flat_param = _FSDP_MODULE_REGISTRY[module_id]
            if flat_param is not None:
                flat_params.append(flat_param)
                seen_ids.add(module_id)
    
    return flat_params


def is_fsdp_managed(module: nn.Module) -> bool:
    """Check if a module is managed by FSDP.
    
    Args:
        module: Module to check
    
    Returns:
        True if module is FSDP-managed
    """
    return hasattr(module, '_is_fsdp_managed') and module._is_fsdp_managed


def clear_fsdp_registry():
    """Clear the global FSDP registry.
    
    Useful for testing or when you want to re-apply FSDP to a model.
    """
    global _FSDP_MODULE_REGISTRY
    _FSDP_MODULE_REGISTRY.clear()


# Export for convenience
__all__ = [
    'fully_shard',
    'get_flat_parameters',
    'is_fsdp_managed',
    'clear_fsdp_registry',
]

