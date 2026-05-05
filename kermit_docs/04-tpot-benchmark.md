# vLLM TPOT Benchmark 实验

实验日期: 2026-04-28
模型: Qwen3-0.6B
引擎: vLLM V1 AsyncLLMEngine (v0.20.0)
GPU: RTX 4060 Ti 16GB

## 实验目的

探索不同 sampling 参数组合命中的 sampling 实现是否有 TPOT 差异。
参考 v0.14.0 中的问题：topp+topk 比 仅 topk 更快。

## Sampling 配置

| 配置名 | SamplingParams | 命中 Code Path |
|--------|---------------|---------------|
| `greedy` | temp=0 | argmax only, 跳过 random sampling |
| `temp_only` | temp=0.7, top_p=1.0, top_k=-1 | temperature + softmax, topk_topp early exit |
| `temp_topp` | temp=0.7, top_p=0.9 | temperature + top_p filtering |
| `temp_topk` | temp=0.7, top_k=50 | temperature + top_k filtering |
| `temp_topp_topk` | temp=0.7, top_p=0.9, top_k=50 | temperature + top_p + top_k |

### Code Path 分派逻辑

```
sampler.py:245
  all_greedy? → argmax only
  all_random? → temperature + topk_topp
  mixed?      → argmax AND random, per-req select

topk_topp_sampler.py:251
  k=None and p=None? → early exit (temp_only)
  batch_size >= 8?   → Triton kernel (TOPK_ENABLED, TOPP_ENABLED compile-time flags)
  batch_size < 8?    → PyTorch fallback (torch.sort)
```

---

## 实验 1: 单请求 TPOT (BF16 vs FP8)

脚本: `bench_tpot.py`
条件: 1 并发请求, 128 output tokens, 3 warmup + 10 iterations

### BF16 结果

| Config | Mean TPOT | Std | vs Greedy |
|--------|-----------|-----|-----------|
| greedy | 4.990ms | 0.003ms | baseline |
| temp_only | 5.037ms | 0.002ms | +0.9% |
| temp_topk | 5.145ms | 0.003ms | +3.1% |
| temp_topp | 5.175ms | 0.002ms | +3.7% |
| temp_topp_topk | 5.205ms | 0.002ms | +4.3% |

### FP8 结果

| Config | Mean TPOT | Std | vs Greedy |
|--------|-----------|-----|-----------|
| greedy | 3.667ms | 0.002ms | baseline |
| temp_only | 3.712ms | 0.002ms | +1.2% |
| temp_topk | 3.809ms | 0.002ms | +3.9% |
| temp_topp | 3.844ms | 0.001ms | +4.8% |
| temp_topp_topk | 3.871ms | 0.002ms | +5.6% |

### BF16 vs FP8

| Config | BF16 | FP8 | FP8 加速 |
|--------|------|-----|---------|
| greedy | 4.990ms | 3.667ms | -26.5% |
| temp_only | 5.037ms | 3.712ms | -26.3% |
| temp_topp | 5.175ms | 3.844ms | -25.7% |
| temp_topk | 5.145ms | 3.809ms | -26.0% |
| temp_topp_topk | 5.205ms | 3.871ms | -25.6% |

---

## 实验 2: 并发 TPOT (BF16, 不同 batch size)

脚本: `bench_tpot_concurrent.py`
条件: 1/4/8/16/32 并发, 64 output tokens, 2 warmup + 5 iterations
目的: 测试 batch_size >= 8 时 Triton kernel 与 batch_size < 8 PyTorch fallback 差异

### TPOT 汇总

| Config | conc=1 | conc=4 | conc=8 | conc=16 | conc=32 |
|--------|--------|--------|--------|---------|---------|
| greedy | 4.98ms | 5.16ms | 5.30ms | 5.67ms | 6.32ms |
| temp_only | 5.03ms | 5.23ms | 5.39ms | 5.78ms | 6.70ms |
| temp_topp | 5.17ms | 5.66ms | 5.60ms | 5.98ms | 6.92ms |
| temp_topk | 5.13ms | 5.48ms | 5.59ms | 5.96ms | 6.87ms |
| temp_topp_topk | 5.20ms | 5.70ms | 5.60ms | 5.98ms | 6.91ms |

### Sampling 开销 (vs greedy, 同并发度)

| Config | conc=1 | conc=4 | conc=8 | conc=16 | conc=32 |
|--------|--------|--------|--------|---------|---------|
| temp_only | +0.9% | +1.3% | +1.8% | +2.0% | +5.9% |
| temp_topp | +3.6% | +9.7% | +5.7% | +5.6% | +9.5% |
| temp_topk | +2.9% | +6.1% | +5.6% | +5.1% | +8.7% |
| temp_topp_topk | +4.2% | +10.4% | +5.7% | +5.5% | +9.3% |

### Triton 分界点分析 (conc=4 [PyTorch] vs conc=8 [Triton])

| Config | conc=4 (PyTorch) | conc=8 (Triton) | Delta |
|--------|-----------------|-----------------|-------|
| greedy | 5.16ms | 5.30ms | +2.7% |
| temp_only | 5.23ms | 5.39ms | +3.2% |
| **temp_topp** | **5.66ms** | **5.60ms** | **-1.1%** |
| temp_topk | 5.48ms | 5.59ms | +2.1% |
| **temp_topp_topk** | **5.70ms** | **5.60ms** | **-1.7%** |

---

## 实验 3: v0.14.0 复现对比

脚本: `bench_tpot_v14.py` (conda env: `vega-14`)
条件: BF16, 1 并发请求, 128 output tokens, 3 warmup + 10 iterations
目的: 在 v0.14.0 上复现 "topp+topk 比 仅 topk 快" 的历史异常

### v0.14.0 单请求 TPOT

| Config | Mean TPOT | Std | vs Greedy |
|--------|-----------|-----|-----------|
| greedy | 4.97ms | 0.00ms | baseline |
| temp_only | 5.01ms | 0.00ms | +0.9% |
| temp_topp | 5.17ms | 0.00ms | +4.0% |
| **temp_topk** | **5.61ms** | **0.01ms** | **+13.0%** |
| temp_topp_topk | 5.20ms | 0.00ms | +4.7% |

### 异常确认

| 对比 | TPOT | Delta |
|------|------|-------|
| temp_topp_topk | 5.200ms | — |
| temp_topk | 5.611ms | **-7.3%** |

**temp_topp_topk 比 temp_topk 快 0.41ms (7.3%)** — 异常复现成功。
加了 top_p 约束反而更快, 不符合直觉。

### v0.14.0 → v0.20.0 跨版本对比

| Config | v0.14.0 | v0.20.0 | 变化 |
|--------|---------|---------|------|
| greedy | 4.97ms | 4.99ms | +0.4% |
| temp_only | 5.01ms | 5.04ms | +0.6% |
| temp_topp | 5.17ms | 5.18ms | +0.2% |
| **temp_topk** | **5.61ms** | **5.15ms** | **-8.2%** |
| temp_topp_topk | 5.20ms | 5.21ms | +0.2% |

**关键洞察**: v0.20.0 的修复不在 topp_topk 路径, 而在 **topk-only** 路径。
temp_topk TPOT 从 5.61ms 降到 5.15ms (降 0.46ms), 其他配置基本不变。
异常的本质是 v0.14.0 的 topk-only 实现效率低下, 而非 topp_topk 异常地快。

---

## 关键发现

### 1. v0.14.0 topp+topk 异常: 根因定位

v0.14.0 中 `temp_topp_topk` (5.20ms) 比 `temp_topk` (5.61ms) 快 7.3%。
跨版本对比证实: 异常源于 **topk-only 实现效率低**, 而非 topp_topk 路径有什么加速。
v0.20.0 将 topk-only TPOT 降了 0.46ms, 修复了这一问题。

### 2. v0.20.0 排序符合预期

v0.20.0 中采样开销排序: `greedy < temp_only < topk < topp < topp_topk`,
完全符合计算复杂度预期, 无异常反转。

### 3. PyTorch top_p fallback 效率低

在 batch_size < 8 时 (conc=4), top_p 开销 +9.7%, 而 top_k 仅 +6.1%。
这是因为 PyTorch fallback 用 `torch.sort()` + 累积概率计算, top_p 比 top_k 计算量更大。

### 4. Triton kernel 消除了 top_p/topk 开销差异

在 batch_size >= 8 时 (conc=8), 三个 filtering 配置的开销几乎一致:
- temp_topp: +5.7%
- temp_topk: +5.6%
- temp_topp_topk: +5.7%

Triton kernel 对不同 sampling 配置的性能差异很小。

### 5. Triton kernel 比 PyTorch fallback 更高效

conc=4 → conc=8, batch_size 翻倍, 但 topp/topp_topk 的 TPOT 反而**下降**:
- temp_topp: 5.66ms → 5.60ms (-1.1%)
- temp_topp_topk: 5.70ms → 5.60ms (-1.7%)

这说明 Triton top_p/topk kernel 比对应的 PyTorch fallback 显著更高效,
足以抵消 batch_size 翻倍带来的额外计算。

### 6. FP8 量化加速 ~26%

所有 sampling 配置下, FP8 TPOT 比 BF16 低 25-26%, 且 sampling 开销比例保持一致。

### 7. CUDA Graph 下方差极低

所有测试的 std < 0.01ms, CUDA Graph 消除了 kernel launch 开销,
使得 TPOT 高度确定性, 便于测量微小差异。

---

## 优化方向

1. **降低 Triton 阈值**: 当前阈值 batch_size=8 可能过高。
   对含 top_p 的请求, 在更小 batch (甚至 1) 使用 Triton kernel 可能更快。
   需要 benchmark 验证 Triton kernel 在 batch_size=1 时的性能。

2. **FlashInfer sampler**: 设置 `VLLM_USE_FLASHINFER_SAMPLER=1` 可启用
   flashinfer 采样, 可能有不同性能特征。

3. **高并发 sampling 开销**: conc=32 时 sampling 开销上升到 ~9%,
   可能值得优化大 batch 下的 sampling 实现。

---

## 脚本和数据文件

| 文件 | 说明 |
|------|------|
| `bench_tpot.py` | 单请求 BF16+FP8 benchmark (v0.20.0) |
| `bench_tpot_concurrent.py` | 并发 BF16 benchmark (v0.20.0) |
| `bench_tpot_v14.py` | 单请求 BF16 benchmark (v0.14.0, 环境需 conda env vega-14) |
| `bench_tpot_results.json` | v0.20.0 单请求原始数据 |
| `bench_tpot_concurrent.json` | v0.20.0 并发原始数据 |
| `bench_tpot_v14_results.json` | v0.14.0 单请求原始数据 |
