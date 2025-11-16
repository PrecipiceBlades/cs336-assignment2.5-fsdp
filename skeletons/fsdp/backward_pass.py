"""Task 4: Backward Pass with Gradient Reduce-Scatter (SKELETON VERSION)

学生需要实现的核心功能：
1. Backward前all-gather parameters（如果forward时resharded了）
2. Backward后reduce-scatter gradients
3. 处理gradient averaging（data parallel）
4. 清零padding gradients

关键学习点：
- 为什么backward前要all-gather？（需要完整参数计算梯度）
- Reduce-scatter vs all-reduce的区别？（memory vs communication）
- 为什么要average gradients？（data parallel标准做法）
- Padding gradients如何处理？（必须清零）
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from fsdp.flat_param import FlatParameter


def reduce_scatter_grads(flat_param: FlatParameter, async_op: bool = False) -> Optional:
    """Reduce-scatter gradients across ranks。
    
    TODO: 实现以下逻辑
    1. 从flat_param._orig_params收集gradients
    2. Flatten并concatenate成full_grad
    3. Pad full_grad到padded_total_numel（padding用0填充）
    4. 如果world_size=1，直接复制gradient
    5. 否则：
       a. 创建local_grad_shard
       b. 调用reduce_scatter_tensor
       c. 除以world_size做averaging（关键！）
       d. 清零padding部分的gradient
    6. 设置flat_param.grad = local_grad_shard
    
    关键问题：
    - 为什么从orig_params收集gradient？（backward在views上计算）
    - 为什么要pad？（reduce-scatter要求uniform sizes）
    - 为什么要averaging？（data parallel标准）
    - 为什么清零padding？（防止影响optimizer update）
    
    提示：
    - Padding gradients必须是0
    - World_size=1时不需要averaging
    - 最后rank的shard可能包含padding
    """
    if async_op:
        raise NotImplementedError("Async not supported")
    
    # TODO: Collect gradients from original parameters
    # full_grad_list = []
    # for orig_param in flat_param._orig_params:
    #     if orig_param.grad is not None:
    #         full_grad_list.append(orig_param.grad.flatten())
    #     else:
    #         full_grad_list.append(torch.zeros_like(orig_param.data.flatten()))
    
    # TODO: Concatenate
    # full_grad = torch.cat(full_grad_list)
    
    # TODO: Handle world_size=1 case
    # if flat_param.world_size == 1:
    #     flat_param.grad = full_grad.clone()
    #     return None
    
    # TODO: Pad gradient
    # if full_grad.numel() < flat_param._padded_total_numel:
    #     padding_size = flat_param._padded_total_numel - full_grad.numel()
    #     full_grad = torch.cat([full_grad, torch.zeros(...)])
    
    # TODO: Reduce-scatter
    # local_grad_shard = torch.empty_like(flat_param.data)
    # reduce_scatter_tensor(output_tensor=local_grad_shard, input_tensor=full_grad)
    
    # TODO: Average (CRITICAL for data parallel!)
    # if flat_param.world_size > 1:
    #     local_grad_shard.div_(flat_param.world_size)
    
    # TODO: Zero out padding in gradient shard
    # shard_start = flat_param._shard_offset
    # shard_end = shard_start + flat_param._shard_numel
    # if shard_end > flat_param._total_numel:
    #     valid_size = flat_param._total_numel - shard_start
    #     local_grad_shard[valid_size:] = 0.0
    
    # TODO: Set gradient
    # flat_param.grad = local_grad_shard
    
    raise NotImplementedError("Students need to implement reduce_scatter_grads")


def create_backward_pre_hook(flat_param: FlatParameter, reshard_after_forward: bool = True):
    """创建backward pre-hook。
    
    TODO: 返回一个hook function，在backward前：
    - 如果forward后resharded了，需要all-gather parameters
    - 更新parameter views
    
    关键：只有reshard_after_forward=True时才需要all-gather
    """
    def backward_pre_hook(module: nn.Module, grad_outputs):
        # TODO: Implement
        # if reshard_after_forward and flat_param._is_sharded:
        #     flat_param.all_gather()
        #     views = flat_param.create_views()
        #     # Update module parameters to use views
        raise NotImplementedError()
    
    return backward_pre_hook


def create_backward_post_hook(flat_param: FlatParameter):
    """创建backward post-hook。
    
    TODO: 返回一个hook function，在backward后：
    - Reduce-scatter gradients
    - Reshard parameters
    
    这个hook在backward完成后调用，确保gradients已计算完毕。
    """
    def post_accumulate_grad_hook(module: nn.Module, grad_input, grad_output):
        # TODO: Implement
        # reduce_scatter_grads(flat_param)
        # flat_param.reshard()
        raise NotImplementedError()
    
    return post_accumulate_grad_hook


def register_backward_hooks(
    module: nn.Module,
    flat_param: FlatParameter,
    reshard_after_forward: bool = True
) -> Tuple:
    """注册backward hooks到module。
    
    TODO: 
    1. 创建pre和post hooks
    2. 使用module.register_full_backward_hook注册
    3. 返回hook handles
    
    提示：使用register_full_backward_hook而不是register_backward_hook
    """
    raise NotImplementedError("Students need to implement register_backward_hooks")

