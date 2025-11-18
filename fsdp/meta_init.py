"""Task 1: Meta Device & Deferred Initialization.

This module implements meta device initialization for FSDP, allowing models to be
constructed without allocating memory, then materialized shard-by-shard on each rank.

Key concepts:
- Meta device: torch.device("meta") creates tensor metadata without storage
- Deferred materialization: Allocate memory only when needed, only on owning rank
- Memory savings: Can construct arbitrarily large models without OOM
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from fsdp.utils import get_rank, get_world_size


def is_meta_device(tensor: torch.Tensor) -> bool:
    """Check if a tensor is on meta device.
    
    Args:
        tensor: Tensor to check
    
    Returns:
        True if tensor is on meta device, False otherwise
    
    Example:
        >>> with torch.device("meta"):
        ...     tensor = torch.randn(10, 10)
        >>> is_meta_device(tensor)
        True
    """
    return tensor.device.type == "meta"


def init_model_on_meta(model_fn: Callable[[], nn.Module]) -> nn.Module:
    """Initialize a model on meta device.
    
    This allows constructing large models without allocating memory.
    
    Args:
        model_fn: Function that returns a model (e.g., lambda: MyModel())
    
    Returns:
        Model with all parameters on meta device
    
    Example:
        >>> model = init_model_on_meta(lambda: nn.Linear(1000, 1000))
        >>> assert all(is_meta_device(p) for p in model.parameters())
        >>> # No memory allocated yet!
        >>> print(model.weight.numel())  # Shape information available
        1000000
    """
    with torch.device("meta"):
        model = model_fn()
    return model


def materialize_meta_tensor(
    meta_tensor: torch.Tensor,
    device: torch.device,
    init_fn: Optional[Callable[[torch.Tensor], None]] = None
) -> torch.Tensor:
    """Materialize a meta tensor on a real device.
    
    Args:
        meta_tensor: Tensor on meta device
        device: Real device to materialize on (e.g., cuda:0, cpu)
        init_fn: Optional initialization function (e.g., torch.nn.init.kaiming_uniform_)
                 If None, initializes to zeros.
    
    Returns:
        Materialized tensor on real device
    
    Example:
        >>> with torch.device("meta"):
        ...     meta_param = torch.randn(10, 10)
        >>> real_param = materialize_meta_tensor(
        ...     meta_param,
        ...     torch.device("cpu"),
        ...     init_fn=lambda t: torch.nn.init.normal_(t, mean=0, std=0.02)
        ... )
        >>> assert real_param.device.type == "cpu"
        >>> assert real_param.shape == meta_param.shape
    """
    # Create empty tensor with same shape and dtype on target device
    materialized = torch.empty_like(meta_tensor, device=device)
    
    # Initialize the tensor
    if init_fn is not None:
        init_fn(materialized)
    else:
        # Default initialization mimics PyTorch's default for Linear layers
        if materialized.ndim >= 2:
            # Weight matrix - use xavier uniform (kaiming uniform)
            torch.nn.init.kaiming_uniform_(materialized, a=(5 ** 0.5))
        else:
            # Bias or 1D tensor - use uniform based on fan-in
            # This matches PyTorch's default for Linear bias
            fan_in = meta_tensor.numel()
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            torch.nn.init.uniform_(materialized, -bound, bound)
    
    return materialized


def materialize_meta_module(
    module: nn.Module,
    device: torch.device,
    init_fn: Optional[Callable[[torch.Tensor], None]] = None
) -> None:
    """Materialize all parameters in a module from meta device to real device.
    
    This function REPLAYS the initialization logic of the original model, ensuring that
    the meta-initialized model has identical parameter values to a directly initialized model.
    
    Key Design Decisions:
    1. **Replay vs Copy**: We replay initialization (not copy from CPU) to support custom
       initialization logic and to avoid temporarily loading the full model.
       
    2. **Initialization Order**: We follow the exact order of BasicsTransformerLM.__init__
       to ensure RNG state is consumed in the same sequence, guaranteeing deterministic results.
       
    3. **Custom Modules**: We detect cs336_basics custom modules (Linear, Embedding, RMSNorm)
       and replay their specific initialization:
       - Linear/Embedding: trunc_normal_ with model-specific std
       - RMSNorm: ones initialization
    
    This modifies the module in-place.
    
    Args:
        module: Module with parameters on meta device
        device: Real device to materialize on (typically CPU for FSDP)
        init_fn: Optional initialization function for non-cs336 modules
    
    Example:
        >>> with torch.device("meta"):
        ...     model = BasicsTransformerLM(**config)
        >>> materialize_meta_module(model, torch.device("cpu"))
        >>> # All parameters now on CPU with correct initialization
        >>> assert all(not p.is_meta for p in model.parameters())
    """
    # Import custom module types from cs336_basics
    try:
        from cs336_basics.model import Linear as CS336Linear, Embedding as CS336Embedding, RMSNorm, BasicsTransformerLM
        has_cs336_types = True
    except ImportError:
        has_cs336_types = False
    
    # Helper function to replay CS336 module initialization
    # This exactly mirrors the initialization in cs336_basics.model
    def init_cs336_module(submodule):
        """Initialize a CS336 custom module by replaying its __init__ logic.
        
        Why replay instead of calling reset_parameters()?
        - CS336Linear and CS336Embedding don't have reset_parameters()
        - We need to ensure exact replication of the original initialization
        - We can control the device (materializing directly to target device)
        """
        if isinstance(submodule, CS336Linear):
            # CS336Linear uses Xavier-like initialization: std = sqrt(2 / (d_in + d_out))
            d_out, d_in = submodule.weight.shape
            std = (2 / (d_in + d_out)) ** 0.5
            weight_init = torch.empty(d_out, d_in, device=device)
            nn.init.trunc_normal_(weight_init, std=std, a=-3*std, b=3*std)
            submodule.weight = nn.Parameter(weight_init, requires_grad=True)
        elif isinstance(submodule, CS336Embedding):
            # CS336Embedding uses trunc_normal_ with std=1.0
            vocab_size, d_model = submodule.weight.shape
            std = 1.0
            weight_init = torch.empty(vocab_size, d_model, device=device)
            nn.init.trunc_normal_(weight_init, std=std, a=-3*std, b=3*std)
            submodule.weight = nn.Parameter(weight_init, requires_grad=True)
        elif isinstance(submodule, RMSNorm):
            # RMSNorm initializes weight to ones (no learnable bias)
            hidden_size = submodule.weight.shape[0]
            submodule.weight = nn.Parameter(torch.ones(hidden_size, device=device))
    
    # Special handling for BasicsTransformerLM to match __init__ order
    # CRITICAL: We must initialize submodules in the EXACT same order as BasicsTransformerLM.__init__
    # This ensures RNG state is consumed in the same sequence, producing identical parameter values
    if has_cs336_types and isinstance(module, BasicsTransformerLM):
        # Follow the exact order in BasicsTransformerLM.__init__:
        
        # 1. token_embeddings (Embedding layer)
        init_cs336_module(module.token_embeddings)
        
        # 2. positional_encoder (RotaryEmbedding - no learnable parameters)
        #    Buffer initialization is handled separately below
        
        # 3. layers (each TransformerBlock in order)
        #    Each block contains: attn (CausalSelfAttention), ffn (FeedForward), ln1, ln2 (RMSNorm)
        #    We traverse with modules() which follows depth-first order
        for layer in module.layers:
            for submodule in layer.modules():
                if submodule is not layer:  # Skip the container itself
                    has_meta = any(p.is_meta for p in submodule.parameters(recurse=False))
                    if has_meta:
                        init_cs336_module(submodule)
        
        # 4. ln_final (RMSNorm)
        init_cs336_module(module.ln_final)
        
        # 5. lm_head (Linear layer for output projection)
        init_cs336_module(module.lm_head)
    else:
        # Generic fallback: use modules() traversal order
        for submodule in module.modules():
            has_meta_params = any(p.is_meta for p in submodule.parameters(recurse=False))
            
            if has_meta_params:
                if has_cs336_types:
                    init_cs336_module(submodule)
                else:
                    # Fallback for non-CS336 modules
                    for param_name, param in submodule.named_parameters(recurse=False):
                        if is_meta_device(param):
                            empty_param = torch.empty_like(param, device=device)
                            setattr(submodule, param_name, nn.Parameter(empty_param, requires_grad=param.requires_grad))
                    
                    if hasattr(submodule, 'reset_parameters'):
                        submodule.reset_parameters()
                    elif init_fn is not None:
                        for param in submodule.parameters(recurse=False):
                            init_fn(param)
    
    # Process buffers
    for submodule in module.modules():
        for buffer_name, buffer in submodule.named_buffers(recurse=False):
            if is_meta_device(buffer):
                materialized = materialize_meta_tensor(buffer, device, None)
                submodule.register_buffer(buffer_name, materialized)


def materialize_shard_only(
    module: nn.Module,
    device: torch.device,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    init_fn: Optional[Callable[[torch.Tensor], None]] = None
) -> None:
    """Materialize only the local shard of parameters for this rank.
    
    This is a key optimization: instead of materializing full parameters then sharding,
    we directly materialize only the shard we need.
    
    Args:
        module: Module with parameters on meta device
        device: Device to materialize shard on
        rank: Current rank (if None, uses get_rank())
        world_size: World size (if None, uses get_world_size())
        init_fn: Optional initialization function
    
    Note:
        This is a simplified version. Full FSDP would need to coordinate
        initialization across ranks to ensure all shards come from the same
        random initialization.
    
    Example:
        >>> with torch.device("meta"):
        ...     model = nn.Linear(100, 100)  # 10,000 parameters
        >>> materialize_shard_only(model, torch.device("cpu"), rank=0, world_size=4)
        >>> # Rank 0 only has ~2,500 parameters materialized
    """
    # Simplified implementation: For now, materialize full parameters
    # In practice, this would be integrated with FlatParameter (Task 2)
    # which handles the actual sharding
    rank = rank if rank is not None else get_rank()
    world_size = world_size if world_size is not None else get_world_size()
    
    # For single rank, just materialize everything
    if world_size == 1:
        materialize_meta_module(module, device, init_fn)
        return
    
    # For multi-rank, we still materialize full params here
    # The actual sharding will happen in FlatParameter (Task 2)
    materialize_meta_module(module, device, init_fn)


# Example usage (for testing):
if __name__ == "__main__":
    # Test 1: Create model on meta device
    print("Test 1: Meta device initialization")
    with torch.device("meta"):
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000)
        )
    
    print(f"Model parameters on meta device: {all(is_meta_device(p) for p in model.parameters())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Memory allocated: {sum(p.storage().size() for p in model.parameters())}")  # Should be 0
    
    # Test 2: Materialize on real device
    print("\nTest 2: Materialize on real device")
    materialize_meta_module(model, torch.device("cpu"))
    print(f"Model parameters on cpu: {all(p.device.type == 'cpu' for p in model.parameters())}")
    print(f"Memory allocated: {sum(p.storage().size() for p in model.parameters())}")  # Should be > 0

