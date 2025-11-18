# FSDP Learning Implementation (CS336-Style)

**✅ Production-ready FSDP implementation following PyTorch FSDP2 API design**

A complete Fully Sharded Data Parallel (FSDP) implementation with Meta Device initialization, achieving perfect numerical equivalence with single-GPU training.

---

## 📝 About This Repository

**Author**: Ruitao Yi ([s1784383@gmail.com](mailto:s1784383@gmail.com))

**Acknowledgements**: This implementation is inspired by the excellent pedagogical approach of [Stanford CS336: Language Modeling from Scratch](https://github.com/stanford-cs336/assignment2-systems). I greatly admire their structured, systems-focused curriculum design. While this follows a similar educational style, it is my independent implementation with extensive documentation and debugging stories.

**Mission**: Help others learn FSDP deeply through:
- 📖 Well-commented code explaining every design decision
- 🐛 Detailed debugging journey documenting real challenges
- 🎯 Interview preparation materials
- ✅ Production-ready implementation with < 0.001% numerical equivalence

**If this helps you**: Please consider starring ⭐ this repo and citing it in your work!

```bibtex
@misc{yi2024fsdp,
  author = {Ruitao Yi},
  title = {FSDP Learning Implementation (CS336-Style)},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/PrecipiceBlades/fsdp-learning-cs336-style}
}
```

---

## 🎯 Key Features

- **Meta Device Support**: Memory-efficient initialization for large models
- **Nested FSDP**: Correct handling of hierarchical model sharding
- **Numerical Equivalence**: < 0.001% error vs single-GPU baseline
- **Memory Efficiency**: 3.9x memory savings per GPU
- **PyTorch FSDP2 Compatible**: Follows official API design

---

## 🎉 Validation Results

### GPT-2 XL (2.1B Parameters) Comparison

**Test**: `tests/test_fsdp_integration.py --config gpt2xl`

All three methods produce **identical training dynamics** with vastly different memory footprints:

| Method | Parameters | Memory per GPU | Total Memory | Loss (Step 0) | Loss (Step 9) | Final Param Sum |
|--------|-----------|----------------|--------------|---------------|---------------|-----------------|
| **Single GPU** | 2.1B | 40.6 GB | 40.6 GB | 10.8519020 | 10.8702612 | 161786.334758 |
| **DDP (8 GPUs)** | 2.1B | 48.5 GB | 387 GB | 10.8519036 | 10.8702625 | 161786.339399 |
| **Meta FSDP (8 GPUs)** | 2.1B | **22.0 GB** | **176 GB** | 10.8519036 | 10.8702623 | 161786.335199 |

**Key Observations**:
- ✅ **Perfect Numerical Equivalence**: All methods produce identical losses (< 0.0001% difference)
- ✅ **2.2× Memory Savings vs DDP**: 48.5 GB → 22.0 GB per GPU
- ✅ **1.8× Memory Savings vs Single GPU**: 40.6 GB → 22.0 GB per GPU  
- ✅ **2.2× Total Memory Savings**: 387 GB → 176 GB across 8 GPUs
- ✅ **Enables Training Larger Models**: Single GPU cannot fit models > 40 GB

**Why DDP uses more memory per GPU than Single GPU?**
- DDP replicates the full model on each GPU
- Additional communication buffers for gradient synchronization
- Slightly higher peak memory during backward pass

**Memory Breakdown (FP32, GPT-2 XL)**:
```
Single GPU:  2.1B params × 4 bytes × 4 (params + grads + Adam states) ≈ 34 GB
DDP:         Same as Single GPU + communication buffers ≈ 48 GB per GPU
Meta FSDP:   2.1B params ÷ 8 GPUs × 4 bytes × 4 ≈ 4.2 GB + overhead ≈ 22 GB per GPU
```

### Small Model (4.7M Parameters) Verification

**Test**: `tests/test_fsdp_integration.py --config small`

| Metric | Single GPU | Meta FSDP (8 GPUs) | Relative Error |
|--------|------------|-------------------|----------------|
| Step 0 Loss | 7.1115722656 | 7.1115728617 | **0.00008%** ✓ |
| Step 4 Loss | 7.0903291702 | 7.0903295875 | **0.00006%** ✓ |
| Final Param Sum | 2286.522773 | 2286.522717 | **0.00002%** ✓ |
| Memory per GPU | 737.84 MB | **187.65 MB** | **3.9× savings** |

```bash
# Run equivalence tests
# Single GPU baseline
uv run tests/test_fsdp_integration.py --mode single --config gpt2xl

# DDP (8 GPUs)
uv run -m torch.distributed.run --nproc_per_node=8 \
    tests/test_fsdp_integration.py --mode ddp --config gpt2xl

# Meta FSDP (8 GPUs) - Best memory efficiency
uv run -m torch.distributed.run --nproc_per_node=8 \
    tests/test_fsdp_integration.py --mode meta_fsdp --config gpt2xl
```

### Multi-GPU Strict Equivalence (Same Data)

**All GPU counts (1/2/4/8) produce identical parameters with the same data!**

| GPU Count | Final Param Sum | Max Diff vs 1 GPU |
|-----------|-----------------|-------------------|
| 1 GPU | 1.880849838256836 | baseline |
| 2 GPUs | 1.880849838256836 | **7.45e-09** ✓ |
| 4 GPUs | 1.880849838256836 | **7.45e-09** ✓ |
| 8 GPUs | 1.880849838256836 | **2.98e-08** ✓ |

**Difference < 3e-08 = Machine Precision = Perfect Equivalence!**

### Real Data Parallel (Different Data per GPU)

**Test**: `tests/test_data_parallel.py`

```bash
$ uv run torchrun --nproc_per_node=4 tests/test_data_parallel.py

Memory sharding: 4.00x
Initial loss: 0.124
Final loss:   0.011
Reduction:    0.113
✅ Training successful!
```

---

## 📚 Core Components

### 1. Meta Device Initialization (`fsdp/meta_init.py`)

**Key Design Decisions**:
- **Replay Initialization**: Replays original model's initialization logic instead of copying
- **Deterministic RNG**: Follows exact `__init__` order to ensure identical parameters across ranks
- **Custom Module Support**: Handles cs336_basics custom modules (Linear, Embedding, RMSNorm)

```python
# Create model on meta device (no memory allocated)
with torch.device("meta"):
    model = BasicsTransformerLM(**config)

# Materialize to CPU, replaying initialization
materialize_meta_module(model, torch.device("cpu"))

# Apply FSDP (only local shard per GPU)
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# Move to GPU (memory-efficient)
model = model.to(device)
```

### 2. FlatParameter (`fsdp/flat_param.py`)

**Critical Bug Fix**: Prevents parameter duplication in nested FSDP

**Problem**: Without proper filtering, nested FSDP would include the same parameter in multiple FlatParameters:
```python
for layer in model.layers:
    fully_shard(layer)  # layer params in FlatParameter #1
fully_shard(model)      # layer params AGAIN in FlatParameter #2 ❌
```

**Solution**: `_is_fsdp_managed_recursively()` checks if child modules are already FSDP-wrapped:
```python
def flatten_module_params(module):
    params = list(module.parameters(recurse=False))  # Direct params
    
    for child in module.named_children():
        if not _is_fsdp_managed_recursively(child):
            params.extend(child.parameters(recurse=True))  # Include non-FSDP children
        # Skip FSDP-managed children (already sharded)
    
    return FlatParameter(params)
```

**Features**:
- Uniform padding for collective ops compatibility
- All-gather and reshard operations
- View management for parameter access

### 3. Forward Pass (`fsdp/forward_pass.py`)

- All-gather parameters before forward
- Create views back to original parameter shapes
- Optional reshard after forward (memory vs communication trade-off)

### 4. Backward Pass (`fsdp/backward_pass.py`)

- Reduce-scatter gradients across ranks
- Gradient averaging (÷ world_size)
- Zero out padding in gradients
- Reshard parameters after backward

### 5. Sharded Optimizer (`fsdp/optimizer.py`)

- Stores optimizer states only for local shard
- Memory: 4N → 4N/W (where W = world_size)
- Zero out padding in parameters after updates

### 6. FSDP2 API (`fsdp/api.py`)

PyTorch-compatible API:
```python
from fsdp.api import fully_shard

# Apply to submodules
for layer in model.layers:
    fully_shard(layer)

# Apply to root (skips already-managed children)
fully_shard(model)
```

---

## 🔑 Key Technical Details

### RNG Determinism

**Challenge**: PyTorch uses different RNGs for CPU vs GPU:
- CPU: Mersenne Twister (MT19937)
- GPU: Philox RNG

**Solution**: Unified CPU initialization across all methods:
```python
# All training modes initialize on CPU first
model = BasicsTransformerLM(**config)  # CPU initialization
model = model.to(device)               # Then move to GPU
```

**Meta Device RNG Management**:
```python
# Save RNG state before creating meta model
rng_state = torch.get_rng_state()

with torch.device("meta"):
    model = BasicsTransformerLM(**config)

# Restore RNG state for deterministic materialization
torch.set_rng_state(rng_state)
materialize_meta_module(model, torch.device("cpu"))
```

### Padding Handling

**Why padding?**
- `all_gather_into_tensor` and `reduce_scatter_tensor` require uniform tensor sizes
- Example: 10 elements, 3 GPUs → shard_size = 4, padded_total = 12

**Padding zeros at three critical points**:
1. **Initialization**: `torch.zeros(padding_size)`
2. **After optimizer step**: Prevent optimizer from updating padding
3. **After reduce-scatter**: Prevent padding gradients from affecting updates

```python
# Zero out padding in gradient shard
if shard_end > total_numel:
    valid_size = total_numel - shard_start
    local_grad_shard[valid_size:] = 0.0

# Zero out padding in parameter shard (after optimizer step)
if shard_end > total_numel:
    valid_size = total_numel - shard_start
    param.data[valid_size:] = 0.0
```

### Gradient Averaging

In data parallel training:
```python
# Reduce-scatter sums gradients from all ranks
reduce_scatter_tensor(output_tensor=local_grad_shard,
                     input_tensor=full_grad)

# Average (only when world_size > 1)
if world_size > 1:
    local_grad_shard.div_(world_size)
```

### Memory Calculation

| Component | Non-FSDP (1 GPU) | FSDP (W GPUs) | Per-GPU Savings |
|-----------|------------------|---------------|-----------------|
| Parameters | N | N/W | W× |
| Gradients | N | N/W | W× |
| Optimizer (Adam) | 2N | 2N/W | W× |
| **Total** | **4N** | **4N/W** | **W×** |

---

## 🧪 Running Tests

### Equivalence Tests

```bash
# Single GPU baseline
uv run tests/test_gpt2xl_equivalence.py --mode single --config small

# Meta FSDP (8 GPUs)
uv run -m torch.distributed.run --nproc_per_node=8 \
    tests/test_gpt2xl_equivalence.py --mode meta_fsdp --config small

# Compare all methods
uv run -m torch.distributed.run --nproc_per_node=8 \
    tests/test_gpt2xl_equivalence.py --mode compare --config small
```

### Multi-GPU Strict Equivalence

```bash
./run_multigpu_test.sh
```

### Data Parallel Training

```bash
uv run torchrun --nproc_per_node=4 tests/test_data_parallel.py
```

### Unit Tests

```bash
# All unit tests
uv run pytest tests/ -v

# Specific modules
uv run pytest tests/test_meta_init.py -v
uv run pytest tests/test_flat_param.py -v
uv run pytest tests/test_forward_pass.py -v
uv run pytest tests/test_backward_pass.py -v
uv run pytest tests/test_optimizer.py -v
```

---

## 📖 Code Structure

```
fsdp/
├── __init__.py          # Package exports
├── config.py            # FSDPConfig
├── utils.py             # Distributed primitives
├── meta_init.py         # Meta device initialization with replay logic
├── flat_param.py        # FlatParameter with duplication prevention
├── forward_pass.py      # All-gather before forward
├── backward_pass.py     # Reduce-scatter after backward
├── optimizer.py         # Sharded optimizer with padding handling
└── api.py               # FSDP2-style API

tests/
├── test_meta_init.py              # Meta device tests
├── test_flat_param.py             # FlatParameter tests
├── test_forward_pass.py           # Forward pass tests
├── test_backward_pass.py          # Backward pass tests
├── test_optimizer.py              # Optimizer tests
├── test_gpt2xl_equivalence.py     # Meta FSDP equivalence
├── test_multigpu_equivalence.py   # Multi-GPU equivalence (same data)
├── test_data_parallel.py          # Data parallel (different data)
└── test_convergence.py            # Single GPU equivalence
```

---

## 🚀 Usage Example

```python
from fsdp.api import fully_shard
from fsdp.optimizer import FSDPOptimizer
from fsdp.meta_init import init_model_on_meta, materialize_meta_module
import torch
import torch.nn as nn

# Option 1: Standard initialization
model = YourTransformer(**config)

# Option 2: Meta device initialization (memory-efficient)
with torch.device("meta"):
    model = YourTransformer(**config)

# Materialize before FSDP
materialize_meta_module(model, torch.device("cpu"))

# Apply FSDP to submodules (inside-out)
for layer in model.layers:
    fully_shard(layer, reshard_after_forward=True)

# Apply FSDP to root module
fully_shard(model)

# Move to GPU
model = model.to("cuda")

# Create optimizer
from fsdp.api import get_flat_parameters
flat_params = get_flat_parameters(model)
optimizer = FSDPOptimizer(
    flat_params,
    optimizer_cls=torch.optim.AdamW,
    lr=1e-4
)

# Train as usual
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## 🎓 Learning Objectives

Students will master:

1. ✅ ZeRO Stage 3 principles and implementation
2. ✅ Parameter sharding and memory calculation
3. ✅ Padding handling for uniform sharding
4. ✅ Collective communications (all-gather, reduce-scatter)
5. ✅ PyTorch autograd hooks (pre-forward, post-forward, post-backward)
6. ✅ Sharded optimizer state management
7. ✅ Meta device initialization for large models
8. ✅ Nested FSDP and parameter duplication prevention
9. ✅ RNG determinism in distributed training

---

## ✨ Project Highlights

1. **Mathematical Correctness**: < 0.001% error vs single-GPU (near machine precision)
2. **Production-Ready**: All core components fully implemented and tested
3. **API Compatible**: Follows PyTorch FSDP2 design patterns
4. **Well-Tested**: Comprehensive unit and integration tests
5. **Well-Documented**: Detailed comments explaining design decisions
6. **Bug-Free**: Critical issues fixed (parameter duplication, RNG determinism)

---

## 🐛 Key Bugs Fixed

### Bug 1: Parameter Duplication in Nested FSDP

**Problem**: Using `module.parameters(recurse=True)` would include parameters from already-FSDP-wrapped child modules, causing:
- Parameters counted 2-3x
- Incorrect all-gather (duplicate parameter values)
- Wrong forward outputs
- Training failure

**Solution**: Implemented `_is_fsdp_managed_recursively()` to skip already-managed children.

**Impact**: Without this fix, the model had 2.37x inflated parameter count!

### Bug 2: RNG Non-Determinism Across CPU/GPU

**Problem**: PyTorch uses different RNGs for CPU (Mersenne Twister) and GPU (Philox), producing different random sequences even with the same seed.

**Solution**: Unified CPU initialization for all training modes.

**Impact**: Ensures bit-exact reproducibility across different FSDP configurations.

### Bug 3: Meta Model RNG Consumption

**Problem**: Creating a meta model consumes RNG state, causing subsequent initialization to diverge.

**Solution**: Save and restore RNG state around meta model creation.

**Impact**: Guarantees deterministic initialization for meta device FSDP.

---

## 📖 References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [PyTorch FSDP2 Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [ZeRO Paper (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP Paper (Zhao et al., 2023)](https://arxiv.org/abs/2304.11277)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

---

## 📝 Additional Documentation

- **`WRITEUP.md`**: Comprehensive technical writeup with implementation details, trade-offs, and interview Q&A
- **`FSDP_DEBUG_JOURNEY.md`**: Complete debugging story with technical learnings (great for interview preparation!)

---

## 🙏 Contributing & Feedback

If you find this helpful or have suggestions for improvement, please:
- ⭐ Star this repository
- 🐛 Open an issue for bugs or questions
- 🔀 Submit a pull request with improvements
- 📧 Email me at s1784383@gmail.com

**Let's learn FSDP together!** This repo is meant to help the community understand distributed training systems better.

---

## 📄 License & Citation

This project is open source under the MIT License.

If you use this implementation or find it helpful for your learning/research, please cite:

```bibtex
@misc{yi2024fsdp,
  author = {Ruitao Yi},
  title = {FSDP Learning Implementation (CS336-Style)},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/PrecipiceBlades/fsdp-learning-cs336-style}},
  note = {Inspired by Stanford CS336: Language Modeling from Scratch}
}
```

---

**Status**: ✅ Production Ready | **Author**: Ruitao Yi | **Last Updated**: November 2024
