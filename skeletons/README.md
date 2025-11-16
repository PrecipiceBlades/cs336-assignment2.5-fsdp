# Skeleton Code for Student Implementation

## 说明

`skeletons/` 目录包含了所有实现的skeleton版本，供学生参考和学习。

完整实现在主目录的 `fsdp/` 中，学生可以：
1. 先阅读skeleton，理解需要实现什么
2. 尝试自己实现
3. 对比完整实现，学习最佳实践
4. 运行tests验证自己的实现

## 学习路径

### Step 1: 理解完整实现
```bash
# 阅读完整实现
fsdp/meta_init.py      # Task 1
fsdp/flat_param.py     # Task 2
fsdp/forward_pass.py   # Task 3
fsdp/backward_pass.py  # Task 4
fsdp/optimizer.py      # Task 5
fsdp/api.py            # FSDP2 API
```

### Step 2: 运行测试理解行为
```bash
# 单元测试
uv run pytest tests/test_meta_init.py -v
uv run pytest tests/test_flat_param.py -v
uv run pytest tests/test_forward_pass.py -v
uv run pytest tests/test_backward_pass.py -v
uv run pytest tests/test_optimizer.py -v

# 等价性测试（最重要！）
uv run python test_full_equivalence.py
```

### Step 3: 尝试自己实现（可选）
```bash
# 复制skeleton到自己的目录
cp -r skeletons/fsdp my_fsdp/

# 实现TODO部分
# ...

# 运行tests验证
uv run pytest tests/ -v
```

## 关键概念清单

学生应该掌握的概念（面试必备）：

### 基础概念
- [ ] ZeRO Stage 1, 2, 3的区别
- [ ] FSDP vs DDP的trade-off
- [ ] Parameter sharding的memory计算（4N → 4N/W）

### 实现细节
- [ ] 为什么需要FlatParameter?
- [ ] Padding的作用和处理方式
- [ ] All-gather和reduce-scatter的时机
- [ ] Autograd hooks的注册和使用
- [ ] Optimizer state sharding

### 高级话题
- [ ] Communication-computation overlap（prefetching）
- [ ] Meta device的作用
- [ ] Reshard after forward的trade-off
- [ ] Mixed precision training
- [ ] CPU offload

## 代码质量要求

学生的实现应该：
1. ✅ 通过所有unit tests
2. ✅ 单GPU FSDP == 单GPU Non-FSDP（严格等价）
3. ✅ 多GPU能正常训练
4. ✅ Memory使用符合预期（4N/W）
5. ✅ 代码清晰，有注释
6. ✅ 理解每个设计决策的原因

## 参考资料

- PyTorch FSDP文档: https://pytorch.org/docs/stable/fsdp.html
- PyTorch FSDP2教程: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- ZeRO论文: https://arxiv.org/abs/1910.02054
- 本实现的VERIFICATION_SUMMARY.md
- 本实现的IMPLEMENTATION_NOTES.md

