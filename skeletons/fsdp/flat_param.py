"""Task 2: FlatParameter Implementation (SKELETON VERSION)

学生需要实现的核心功能：
1. 将多个参数flatten成一个tensor
2. Shard这个tensor到不同ranks
3. 实现all-gather和reshard操作
4. 处理padding以支持uniform shards

关键学习点：
- 为什么需要FlatParameter？（减少通信次数，提高效率）
- 如何计算shard范围？（考虑padding）
- All-gather和reduce-scatter的时机？
- 如何创建views回到原始形状？
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from fsdp.utils import (
    get_rank,
    get_world_size,
    flatten_params,
    all_gather_tensor,
)


class FlatParameter(nn.Parameter):
    """将多个参数flatten并shard的Parameter子类。
    
    TODO: 学生需要实现：
    1. __new__ 和 __init__ 方法
    2. all_gather() 方法
    3. reshard() 方法
    4. create_views() 方法
    
    提示：
    - 继承nn.Parameter需要override __new__
    - 需要处理padding以确保uniform shard sizes
    - World_size=1是special case（不需要actual sharding）
    """
    
    def __new__(cls, params: List[nn.Parameter], rank: Optional[int] = None, world_size: Optional[int] = None):
        """创建FlatParameter。
        
        TODO: 实现以下逻辑
        1. Flatten所有params into single tensor
        2. 计算uniform shard size (考虑padding)
        3. 提取当前rank的shard
        4. 用shard data创建Parameter instance
        5. 存储metadata (total_numel, padded_numel, shard_offset等)
        
        关键问题：
        - 为什么需要padding？（all_gather要求uniform sizes）
        - 如何计算shard_size？（ceiling division）
        """
        rank = rank if rank is not None else get_rank()
        world_size = world_size if world_size is not None else get_world_size()
        
        # TODO: Flatten parameters
        # flat_param_full = flatten_params(params)
        # total_numel = flat_param_full.numel()
        
        # TODO: Calculate uniform shard size with padding
        # shard_size = (total_numel + world_size - 1) // world_size
        # padded_total_numel = shard_size * world_size
        
        # TODO: Pad if necessary
        # if padded_total_numel > total_numel:
        #     ...
        
        # TODO: Extract local shard for this rank
        # start = rank * shard_size
        # end = start + shard_size
        # local_shard_data = flat_param_full[start:end].clone().detach()
        
        # TODO: Create Parameter instance
        # instance = super().__new__(cls, local_shard_data, requires_grad=True)
        
        # TODO: Store metadata for __init__
        # instance._init_params = params
        # instance._init_rank = rank
        # ...
        
        raise NotImplementedError("Students need to implement __new__")
    
    def __init__(
        self,
        params: List[nn.Parameter],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        """初始化FlatParameter。
        
        TODO: 从__new__中的temporary attributes提取metadata
        - self.rank, self.world_size
        - self._total_numel, self._padded_total_numel
        - self._shard_offset, self._shard_numel
        - self._param_shapes (用于create_views)
        
        并初始化：
        - self._full_param = None
        - self._is_sharded = True
        """
        raise NotImplementedError("Students need to implement __init__")
    
    @property
    def local_shard(self) -> torch.Tensor:
        """返回local shard。
        
        TODO: 
        - 如果is_sharded: return self.data
        - 否则: return shard from _full_param
        """
        raise NotImplementedError()
    
    @property
    def full_param(self) -> Optional[torch.Tensor]:
        """返回full parameter (only after all-gather)."""
        return self._full_param
    
    def all_gather(self) -> torch.Tensor:
        """All-gather parameter shards from all ranks。
        
        TODO: 实现all-gather逻辑
        1. 如果已经gathered，直接返回cached _full_param
        2. 如果world_size=1，直接用self.data（特殊处理！）
        3. 否则：
           a. 分配padded_full_param (size = _padded_total_numel)
           b. 调用all_gather_tensor
           c. Slice to get unpadded version
           d. 设置_is_sharded = False
        
        关键问题：
        - 为什么world_size=1要特殊处理？（避免clone，保持同一tensor）
        - 为什么要slice？（去掉padding）
        """
        raise NotImplementedError("Students need to implement all_gather")
    
    def reshard(self) -> None:
        """Reshard回local shard only。
        
        TODO: 释放_full_param，保留local shard
        
        关键警告：
        - 不要从_full_param复制回self.data！
        - Optimizer直接更新self.data
        - 复制会覆盖optimizer的更新
        """
        raise NotImplementedError("Students need to implement reshard")
    
    def create_views(self) -> List[torch.Tensor]:
        """创建views回到原始parameter shapes。
        
        TODO: 
        1. 使用_full_param（如果available）或self.data
        2. 按照_param_shapes切分成views
        3. 返回views列表
        
        关键：
        - Views共享存储！修改view = 修改flat_param
        """
        raise NotImplementedError("Students need to implement create_views")


def flatten_module_params(
    module: nn.Module,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> FlatParameter:
    """将module的所有parameters flatten成一个FlatParameter。
    
    TODO:
    1. 收集module.parameters(recurse=True)
    2. 创建FlatParameter
    
    注意：recurse=True很重要！
    """
    raise NotImplementedError("Students need to implement flatten_module_params")

