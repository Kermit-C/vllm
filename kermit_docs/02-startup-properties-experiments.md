# vLLM AsyncLLMEngine 启动属性实验

实验日期: 2026-04-28
模型: Qwen3-0.6B
引擎: vLLM V1 AsyncLLMEngine

## 实验 1: BF16 默认模式

脚本: `test_async_engine.py`

```python
engine_args = AsyncEngineArgs(
    model=".huggingface/Qwen3-0.6B",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)
```

### 启动属性

| 属性 | 值 |
|---|---|
| 引擎版本 | V1 LLM engine (v0.20.0) |
| 模型架构 | Qwen3ForCausalLM |
| 数据类型 | bfloat16 |
| Attention 后端 | FLASH_ATTN (FlashAttention v2) |
| 可选 Attention 后端 | FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION |
| RMS Norm 后端 | native |
| 编译模式 | VLLM_COMPILE (torch.compile + inductor) |
| CUDA Graph 模式 | FULL_AND_PIECEWISE |
| CUDA Graph capture | PIECEWISE 35个图 (largest=256), FULL 19个图 (largest=128) |
| CUDA Graph 显存 | 0.29 GiB, 耗时 1s |
| torch.compile | 16.34s (首次冷启动), 缓存到 `~/.cache/vllm/torch_compile_cache/` |
| 调度策略 | Asynchronous scheduling + Chunked prefill |
| max_num_batched_tokens | 2048 |
| 并行 | TP=1, PP=1, DP=1 |
| 通信后端 | NCCL |
| KV Cache | 114,160 tokens / 12.19 GiB |
| 最大并发 | 55.74x (以 2048 tokens/request 计) |
| Prefix Caching | 已启用 |
| 自定义融合 | fuse_norm_quant=False, fuse_act_quant=False |
| 量化 | None |
| Speculative Decoding | None |
| 模型加载显存 | 1.12 GiB |
| 模型加载耗时 | 0.65s (weights 0.22s) |
| 总启动耗时 | 25.48s (其中 torch.compile 16.34s) |

### SM 数限制

```
Not enough SMs to use max_autotune_gemm mode
```

RTX 4060 Ti SM 数不足，torch.compile 的 GEMM autotuning 降级。

---

## 实验 2: FP8 量化模式

脚本: `test_fp8_engine.py`

```python
engine_args = AsyncEngineArgs(
    model=".huggingface/Qwen3-0.6B",
    quantization="fp8",
    kv_cache_dtype="fp8",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)
```

### 启动属性

| 属性 | 值 |
|---|---|
| quantization | fp8 |
| kv_cache_dtype | fp8 |
| Attention 后端 | FLASHINFER |
| 可选 Attention 后端 | FLASHINFER, TRITON_ATTN (FP8 KV cache 不支持 FLASH_ATTN) |
| Linear Kernel | CutlassFP8ScaledMMLinearKernel |
| 模型加载显存 | 0.71 GiB |
| KV Cache | 228,464 tokens / 12.2 GiB |
| 最大并发 | 111.55x |
| CUDA Graph 显存 | 0.48 GiB, 耗时 1s |
| torch.compile | 3.89s (命中缓存) |
| FlashInfer warmup | ~26s (首次 JIT 编译) |
| 总启动耗时 | 32.51s |

### FP8 注意事项

- 动态 FP8 量化，无需预量化模型或校准数据
- 启动时有 3 个 warning：
  - checkpoint 未提供 q scaling factor，fallback 到 k_scale
  - KV cache scaling factor 默认 1.0 (未校准，可能有精度损失)
  - uncalibrated q_scale/prob_scale，可能影响 attention 精度
- FP8 KV cache 强制使用 flashinfer 后端，首次需 JIT 编译，需要 `ninja` 在 PATH

---

## 对比总结

| 属性 | BF16 默认 | FP8 量化 | 变化 |
|---|---|---|---|
| 模型显存 | 1.12 GiB | 0.71 GiB | **-37%** |
| KV Cache tokens | 114,160 | 228,464 | **+100%** |
| 最大并发 | 55.74x | 111.55x | **+100%** |
| Attention 后端 | FLASH_ATTN v2 | FLASHINFER | FP8 KV cache 需 flashinfer |
| CUDA Graph 显存 | 0.29 GiB | 0.48 GiB | +66% |
| 总启动耗时 (冷启动) | ~25s | ~33s | +32% |
| 精度 | 完整 | 有损失 (scale 未校准) | 需校准数据 |
| 依赖 | 无额外依赖 | 需要 ninja (flashinfer JIT) | 需预配置 |

---

## 关键发现

1. **FP8 KV cache 是主要收益点**: token 容量翻倍，因为 fp8 只占 bf16 一半字节
2. **FP8 权重量化省显存 37%**: 模型从 1.12 GiB 降到 0.71 GiB
3. **FP8 强制 flashinfer 后端**: 不再走 flash_attn，因为 flash_attn 不支持 fp8 KV cache
4. **首次启动 flashinfer JIT 很慢**: 约 26s，后续有缓存
5. **动态 FP8 无需校准**: 但 scale factor 默认 1.0，精度不如校准后的静态 FP8
6. **torch.compile 首次慢，有缓存后快**: 冷启动 16s，缓存命中后 4s
7. **CUDA Graph memory profiling**: 默认启用 (v0.21.0+)，会预留额外显存给 CUDA Graph
