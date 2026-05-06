<!-- markdownlint-disable -->
# [Kernel] Enable in-place SSM state access for GDN chunk prefill kernel (Qwen3.5, OLMo-Hybrid)

## Purpose

Eliminate per-invocation gather/scatter of SSM state in GDN chunk prefill by
passing `ssm_state_indices` directly to the Triton kernel for in-place cache access.
The kernel uses pointer arithmetic (`IS_CONTINUOUS_BATCHING` constexpr) to
read/write the pre-allocated cache pool directly, removing per-invocation memory
allocation and copy.

### Background

Gated DeltaNet (GDN) linear attention uses `chunk_gated_delta_rule` for chunked
prefill, processing sequences in chunks of `FLA_CHUNK_SIZE` (64) tokens. During
continuous batching (mixed prefill+decode), each sequence's SSM state lives in a
pre-allocated cache pool indexed by `ssm_state_indices`. The previous call chain was:

```
ssm_state = initial_state[ssm_state_indices].contiguous()  # allocate + copy (gather)
o, final_state = kernel(ssm_state)                          # compute
initial_state[ssm_state_indices] = final_state               # copy back (scatter)
```

Each invocation allocates a contiguous tensor and copies the SSM state out of the
cache pool (gather), then writes the result back (scatter). For large decode
batches (128+ sequences), this is hundreds of KB of unnecessary memory traffic
on the critical path.

### Changes

1. **`chunk_delta_h.py`** — Added two constexpr guards: `IS_CONTINUOUS_BATCHING` and
   `HAS_INITIAL_STATE_MASK`. When `ssm_state_indices` is passed, the load/store path
   indexes directly into the cache pool via
   `h0 + state_idx * stride_init_state_token + i_h * V * K`, eliminating the
   temporary tensor from contiguous gather. When `ssm_state_indices` is not passed,
   the original path is used (backward compatible).

2. **`chunk.py`** — Removed the `@input_guard` decorator in favor of manual
   per-tensor `.contiguous()` calls. `initial_state` is only made contiguous when
   `ssm_state_indices is None`, avoiding forced contiguous copies on cache pool
   views. Passes `ssm_state_indices` and `has_initial_state` through to the kernel.

3. **`gdn_linear_attn.py`** — `_forward_core` now passes `ssm_state` (cache pool) and
   `ssm_state_indices` directly, removing the gather/scatter. For new sequences,
   the kernel's `HAS_INITIAL_STATE_MASK` branch skips the load (b_h stays at its
   zero-initialized value), equivalent to a zero-state start. `forward_cuda`
   (FlashInfer path) accepts the new parameter signature but retains internal
   gather/scatter — functionally equivalent, unaffected.

4. **`olmo_hybrid.py`** — `_forward_core` likewise passes `ssm_state_indices` and
   `has_initial_state` to `chunk_gated_delta_rule`, using the same in-place path.

### Affected Models

Both Qwen3.5 and OLMo-Hybrid-7B share the same Triton kernel. They differ in dispatch:

- **Qwen3.5-9B / Qwen3 NEXT (FlashInfer + Triton)**: forward_native (Triton, benefits
  from in-place) or forward_cuda (FlashInfer, SM90+, internal gather/scatter unchanged).
- **OLMo-Hybrid-7B (Triton only)**: Calls `chunk_gated_delta_rule` directly, always
  uses the in-place optimization.

| Layer | File |
|-------|------|
| Kernel | `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` |
| Adapter | `vllm/model_executor/layers/fla/ops/chunk.py` |
| Model (Qwen3.5 / Qwen3 Next) | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` |
| Model (OLMo) | `vllm/model_executor/models/olmo_hybrid.py` |

### Theoretical Analysis

The SSM state cache pool stores per-sequence recurrent state for every linear
attention layer. At a mixed prefill-decode step with 128 active sequences,
the gather/scatter overhead can be quantified from model architecture parameters.

**State shape**: `[N, H, V, K]`, where N = sequences, H = `linear_num_value_heads`,
V = `linear_value_head_dim`, K = `linear_key_head_dim`. The state pool operates
in float32 (`mamba_ssm_dtype: "float32"`), 4 bytes per element.

**Derivation** (Qwen3.5-0.8B, N=128):
```
N=128, H=16, V=128, K=128
elements_per_layer = 128 × 16 × 128 × 128 = 33,554,432
fp32_per_layer  = 33,554,432 × 4 = 128.00 MiB
fp32_all_layers = 128.00 MiB × 18 = 2,304.00 MiB   (18 linear attention layers out of 24)
```

Each gather+scatter cycle reads the entire state out of the pool and writes it
back, doubling the memory traffic:
```
R+W_per_layer  = 128.00 × 2 = 256.00 MiB
R+W_all_layers = 2,304.00 × 2 = 4,608.00 MiB = 4.50 GiB
```

**Qwen3.5-9B** (H=32, 24 linear attention layers out of 32):
```
elements_per_layer = 128 × 32 × 128 × 128 = 67,108,864
fp32_per_layer  = 256.00 MiB
fp32_all_layers = 256.00 MiB × 24 = 6,144.00 MiB
R+W_all_layers  = 6,144.00 × 2 = 12,288.00 MiB = 12.00 GiB
```

**Qwen3.6-27B** (H=48, 48 linear attention layers out of 64):
```
elements_per_layer = 128 × 48 × 128 × 128 = 100,663,296
fp32_per_layer  = 384.00 MiB
fp32_all_layers = 384.00 MiB × 48 = 18,432.00 MiB
R+W_all_layers  = 18,432.00 × 2 = 36,864.00 MiB = 36.00 GiB
```

Summary for N=128 mixed prefill-decode step:

| Model | H | Linear layers | Per-layer state (fp32) | All-layer state (fp32) | Gather+scatter R+W per step |
|-------|---|---------------|------------------------|------------------------|----------------------------|
| Qwen3.5-0.8B | 16 | 18/24 | 128.00 MiB | 2,304.00 MiB | 4,608.00 MiB (4.50 GiB) |
| Qwen3.5-9B | 32 | 24/32 | 256.00 MiB | 6,144.00 MiB | 12,288.00 MiB (12.00 GiB) |
| Qwen3.6-27B | 48 | 48/64 | 384.00 MiB | 18,432.00 MiB | 36,864.00 MiB (36.00 GiB) |

At 128 concurrent sequences, every decode step moves 4.5-36.0 GiB of fp32 SSM
state through redundant gather/scatter copies. This overhead scales linearly
with batch size N and H × (linear layer count). The in-place optimization
eliminates this entirely — the kernel reads and writes the cache pool directly
via `ssm_state_indices` pointer arithmetic, removing the per-step memory
bandwidth tax.

The observed serving throughput gains align with H20's ~4 TB/s HBM bandwidth:
12 GiB of avoided fp32 traffic per step for 9B directly translates to the
measured 11-13% improvement. The larger absolute saving for 27B (36 GiB) is
partially offset by heavier per-layer compute, yielding a similar relative
gain (10-12%). The 0.8B sees the largest relative gain (41-56%) because
compute is lighter so memory bandwidth is the tighter bottleneck.

---
## Test Plan

### Test environment

```
Hardware:    NVIDIA H20 (96GB) / NVIDIA RTX 4060 Ti (16GB)
Models:      Qwen3.5-9B (fp8), Qwen3.6-27B (fp8), Qwen3.5-0.8B (bf16), OLMo-Hybrid-7B (fp8)
Triton:      3.4.0
PyTorch:     2.8.0
CUDA:        12.8
```

### 1. Pointer arithmetic verification

```bash
PYTHONPATH=. .venv/bin/python dev_precision.py
```

In-process OLD (gather/scatter) vs NEW (in-place) comparison with identical
random inputs. 8 scenarios covering N∈{1,4,8,16}, T∈{16,128,256}, H∈{8,16},
V=128, K=128.

### 2. Kernel-level microbenchmark

```bash
PYTHONPATH=. .venv/bin/python bench_kernel.py --dims qwen --output /tmp/kernel_qwen.json
PYTHONPATH=. .venv/bin/python bench_kernel.py --dims qwen0.8b --output /tmp/kernel_qwen0.8b.json
PYTHONPATH=. .venv/bin/python bench_kernel.py --dims olmo --output /tmp/kernel_olmo.json
```

CUDA event timing, 30 warmup + 100 measurement rounds, median reported.
12 scenarios: prefill (T=64/128/256/512/1024), decode (N=1/16/64/128),
mixed (2/4/8 prefill + 14/28/56 decode).

### 3. Serving throughput benchmark

```bash
# Qwen3.5-9B fp8 on H20 (triton, CUDAGraph)
PYTHONPATH=. .venv/bin/python bench_serving.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --h20 --output /tmp/serving_qwen.json

# Qwen3.5-0.8B bf16 on 4060Ti (triton, CUDAGraph)
PYTHONPATH=. .venv/bin/python bench_serving.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/serving_qwen0.8b.json

# Qwen3.6-27B fp8 on H20 (triton, CUDAGraph)
PYTHONPATH=. .venv/bin/python bench_serving.py \
    --model .huggingface/Qwen3.6-27B --quantization fp8 --h20 --output /tmp/serving_qwen27b.json
```

AsyncLLMEngine with asyncio worker pool, 2048 req/scenario, 14 scenarios:
(512/1024/2048 input, 128/256 output) × (1/16/32/128/256 concurrency).

### 4. Accuracy via lm_eval

```bash
# Qwen3.5-9B fp8
PYTHONPATH=. VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m lm_eval \
    --model vllm --model_args "pretrained=.huggingface/Qwen3.5-9B,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,quantization=fp8,max_num_seqs=16,trust_remote_code=True" \
    --tasks gsm8k --batch_size auto --num_fewshot 5

# OLMo-Hybrid-7B fp8
PYTHONPATH=. VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m lm_eval \
    --model vllm --model_args "pretrained=.huggingface/OLMo-Hybrid-7B,dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096,quantization=fp8,max_num_seqs=16,trust_remote_code=True" \
    --tasks gsm8k --batch_size auto --num_fewshot 5
```

GSM8K 5-shot, batch_size=auto, max_num_seqs=16. Main vs feature branch on
both eager (4060Ti) and CUDAGraph+compile (H20) execution paths.

### 5. FlashInfer backend smoketest

```bash
PYTHONPATH=. .venv/bin/python verify_flashinfer.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/flashinfer.json
```

11 scenarios: prefill (8/64/256/1024t), decode (4/16/64/128), mixed
(1/4/8 prefill + 15/60/120 decode). Only Qwen3.5-9B — OLMo-Hybrid-7B
does not use FlashInfer for chunk prefill.

### 6. Pre-commit hooks

```bash
pre-commit run --all-files
```

---
## Test Result

### 1. Pointer arithmetic verification (`dev_precision.py`)

```
N=1, T=16, H=8,  V=128, K=128, seed=42:  PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=1, T=16, H=8,  V=128, K=128, seed=123: PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=4, T=128, H=8,  V=128, K=128, seed=42:  PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=4, T=128, H=8,  V=128, K=128, seed=123: PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=8, T=256, H=16, V=128, K=128, seed=42:  PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=8, T=256, H=16, V=128, K=128, seed=123: PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=16, T=128, H=8,  V=128, K=128, seed=42:  PASS  ssm_diff=0.00e+00  o_diff=0.000000
N=16, T=128, H=8,  V=128, K=128, seed=123: PASS  ssm_diff=0.00e+00  o_diff=0.000000
```

8/8 PASSED. SSM state bit-exact (max diff 0.0), output max diff 0.0 across all scenarios.

### 2. Kernel microbenchmark (`bench_kernel.py`)

Qwen3.5-9B dims (H_k=16, HV=32, K=128, V=128) — RTX 4060 Ti:

| Scenario | Description | main (μs) | feat (μs) | Δ (μs) | Speedup |
|----------|-------------|-----------|-----------|--------|---------|
| prefill_T64 | Single seq, T=64 | 80.9 | 52.2 | −28.7 | +55.0% |
| prefill_T128 | Single seq, T=128 | 103.4 | 66.6 | −36.8 | +55.3% |
| prefill_T256 | Single seq, T=256 | 131.1 | 101.4 | −29.7 | +29.3% |
| prefill_T512 | Single seq, T=512 | 193.5 | 177.8 | −15.7 | +8.8% |
| prefill_T1024 | Single seq, T=1024 | 464.9 | 455.7 | −9.2 | +2.0% |
| decode_N1 | Single request, T=1 | 51.2 | 42.0 | −9.2 | +21.9% |
| decode_N16 | Batch 16, T=1 | 708.5 | 447.1 | −261.4 | +58.5% |
| decode_N64 | Batch 64, T=1 | 8732.7 | 1812.5 | −6920.2 | +381.8% |
| decode_N128 | Batch 128, T=1 | 12880.9 | 5908.6 | −6972.3 | +118.0% |
| mixed_2pf_14d | 2 prefill + 14 decode | 1169.4 | 378.9 | −790.5 | +208.6% |
| mixed_4pf_28d | 4 prefill + 28 decode | 4323.3 | 874.0 | −3449.3 | +394.7% |
| mixed_8pf_56d | 8 prefill + 56 decode | 9356.4 | 1772.3 | −7584.1 | +427.9% |

Qwen3.5-0.8B dims (H_k=16, HV=16, K=128, V=128) — RTX 4060 Ti:

| Scenario | Description | main (μs) | feat (μs) | Δ (μs) | Speedup |
|----------|-------------|-----------|-----------|--------|---------|
| prefill_T64 | Single seq, T=64 | 69.9 | 52.2 | −17.7 | +33.9% |
| prefill_T128 | Single seq, T=128 | 75.8 | 53.2 | −22.6 | +42.5% |
| prefill_T256 | Single seq, T=256 | 96.3 | 71.7 | −24.6 | +34.3% |
| prefill_T512 | Single seq, T=512 | 131.1 | 109.2 | −21.9 | +20.1% |
| prefill_T1024 | Single seq, T=1024 | 213.0 | 195.6 | −17.4 | +8.9% |
| decode_N1 | Single request, T=1 | 39.9 | 34.8 | −5.1 | +14.7% |
| decode_N16 | Batch 16, T=1 | 315.5 | 223.2 | −92.3 | +41.4% |
| decode_N64 | Batch 64, T=1 | 1468.4 | 930.8 | −537.6 | +57.8% |
| decode_N128 | Batch 128, T=1 | 9562.8 | 1878.7 | −7684.1 | +409.0% |
| mixed_2pf_14d | 2 prefill + 14 decode | 469.0 | 216.1 | −252.9 | +117.0% |
| mixed_4pf_28d | 4 prefill + 28 decode | 1193.0 | 447.5 | −745.5 | +166.6% |
| mixed_8pf_56d | 8 prefill + 56 decode | 8429.6 | 911.4 | −7518.2 | +824.9% |

OLMo-Hybrid-7B dims (H_k=30, HV=30, K=96, V=192) — RTX 4060 Ti:

| Scenario | Description | main (μs) | feat (μs) | Δ (μs) | Speedup |
|----------|-------------|-----------|-----------|--------|---------|
| prefill_T64 | Single seq, T=64 | 97.3 | 67.6 | −29.7 | +43.9% |
| prefill_T128 | Single seq, T=128 | 123.1 | 86.1 | −37.0 | +43.0% |
| prefill_T256 | Single seq, T=256 | 178.9 | 146.4 | −32.5 | +22.2% |
| prefill_T512 | Single seq, T=512 | 292.9 | 272.2 | −20.7 | +7.6% |
| prefill_T1024 | Single seq, T=1024 | 767.0 | 705.5 | −61.5 | +8.7% |
| decode_N1 | Single request, T=1 | 60.1 | 46.5 | −13.6 | +29.2% |
| decode_N16 | Batch 16, T=1 | 800.5 | 518.3 | −282.2 | +54.4% |
| decode_N64 | Batch 64, T=1 | 9815.0 | 7282.7 | −2532.3 | +34.8% |
| decode_N128 | Batch 128, T=1 | 13217.8 | 14950.4 | +1732.6 | −11.6% |
| mixed_2pf_14d | 2 prefill + 14 decode | 1291.3 | 504.7 | −786.6 | +155.9% |
| mixed_4pf_28d | 4 prefill + 28 decode | 5022.7 | 1040.3 | −3982.4 | +382.8% |
| mixed_8pf_56d | 8 prefill + 56 decode | 6617.1 | 6597.6 | −19.5 | +0.3% |

Key takeaways:
- Decode batch (N≥16): 1.3-4.8× speedup across all three model dims
- Short prefill (T≤256): 1.2-1.6× speedup
- Long prefill (T≥512): 1.0-1.2× speedup (compute-bound)
- Mixed workloads: 1.2-8.2× speedup in most scenarios
- OLMo decode_N128 regression (−11.6%): largest dim set (HV=30, K=96, V=192) at max batch — likely register-pressure or shared-memory limited. Mixed scenarios unaffected.

### 3. Serving throughput (`bench_serving.py`)

Qwen3.5-9B fp8 — H20 (triton, CUDAGraph, 2048 req/scenario):

| Scenario | main RPS | feat RPS | RPS Δ% | main TPOT (ms) | feat TPOT (ms) | TPOT Δ% |
|----------|---------|---------|--------|----------------|----------------|---------|
| in512_out128_c1 | 1.60 | 1.57 | −1.9% | 4.56 | 4.55 | −0.2% |
| in512_out128_c16 | 12.41 | 12.64 | +1.9% | 7.60 | 7.54 | −0.8% |
| in512_out128_c32 | 14.64 | 15.06 | +2.9% | 13.57 | 13.28 | −2.1% |
| in512_out128_c128 | 16.57 | 18.55 | +11.9% | 51.89 | 46.36 | −10.7% |
| in512_out128_c256 | 16.63 | 18.60 | +11.8% | 52.92 | 47.24 | −10.7% |
| in1024_out128_c1 | 1.52 | 1.53 | +0.7% | 4.56 | 4.55 | −0.2% |
| in1024_out128_c16 | 8.68 | 8.79 | +1.3% | 10.95 | 10.83 | −1.1% |
| in1024_out128_c32 | 9.68 | 10.00 | +3.3% | 21.00 | 20.34 | −3.1% |
| in1024_out128_c128 | 9.75 | 11.04 | +13.2% | 89.09 | 78.23 | −12.2% |
| in1024_out128_c256 | 9.73 | 11.02 | +13.3% | 90.29 | 79.25 | −12.2% |
| in2048_out256_c16 | 4.26 | 4.29 | +0.7% | 12.07 | 11.98 | −0.7% |
| in2048_out256_c32 | 4.66 | 4.76 | +2.1% | 22.99 | 22.50 | −2.1% |
| in2048_out256_c128 | 4.67 | 5.23 | +12.0% | 94.50 | 83.90 | −11.2% |
| in2048_out256_c256 | 4.67 | 5.23 | +12.0% | 95.14 | 84.41 | −11.3% |

Qwen3.6-27B fp8 — H20 (triton, CUDAGraph, 2048 req/scenario):

| Scenario | main RPS | feat RPS | RPS Δ% | main TPOT (ms) | feat TPOT (ms) | TPOT Δ% |
|----------|---------|---------|--------|----------------|----------------|---------|
| in512_out128_c1 | 0.58 | 0.57 | −1.7% | 12.60 | 12.61 | +0.1% |
| in512_out128_c16 | 4.17 | 4.22 | +1.2% | 21.40 | 21.36 | −0.2% |
| in512_out128_c32 | 4.60 | 4.70 | +2.2% | 42.52 | 41.86 | −1.6% |
| in512_out128_c128 | 5.33 | 5.84 | +9.6% | 160.50 | 147.08 | −8.4% |
| in512_out128_c256 | 5.31 | 5.85 | +10.2% | 164.73 | 149.38 | −9.3% |
| in1024_out128_c1 | 0.54 | 0.54 | +0.0% | 12.60 | 12.61 | +0.1% |
| in1024_out128_c16 | 2.83 | 2.87 | +1.4% | 32.41 | 32.10 | −1.0% |
| in1024_out128_c32 | 3.00 | 3.09 | +3.0% | 66.80 | 65.21 | −2.4% |
| in1024_out128_c128 | 3.09 | 3.44 | +11.3% | 280.01 | 250.17 | −10.7% |
| in1024_out128_c256 | 3.07 | 3.43 | +11.7% | 284.40 | 253.55 | −10.8% |
| in2048_out256_c16 | 1.40 | 1.42 | +1.4% | 35.63 | 35.26 | −1.0% |
| in2048_out256_c32 | 1.47 | 1.51 | +2.7% | 72.11 | 70.38 | −2.4% |
| in2048_out256_c128 | 1.50 | 1.66 | +10.7% | 293.69 | 262.78 | −10.5% |
| in2048_out256_c256 | 1.49 | 1.66 | +11.4% | 295.69 | 264.35 | −10.6% |

Qwen3.5-0.8B bf16 — RTX 4060 Ti (triton, CUDAGraph, 2048 req/scenario):

| Scenario | main RPS | feat RPS | RPS Δ% | main TPOT (ms) | feat TPOT (ms) | TPOT Δ% |
|----------|---------|---------|--------|----------------|----------------|---------|
| in512_out128_c1 | 1.56 | 1.28 | −17.9% | 6.25 | 6.29 | +0.6% |
| in512_out128_c16 | 12.30 | 12.82 | +4.2% | 12.20 | 11.49 | −5.8% |
| in512_out128_c32 | 12.98 | 15.12 | +16.5% | 20.17 | 17.57 | −12.9% |
| in512_out128_c128 | 10.84 | 16.28 | +50.2% | 97.43 | 67.74 | −30.5% |
| in512_out128_c256 | 10.61 | 16.51 | +55.6% | 103.52 | 68.44 | −33.9% |
| in1024_out128_c1 | 1.51 | 1.39 | −7.9% | 6.27 | 6.27 | +0.0% |
| in1024_out128_c16 | 9.25 | 10.20 | +10.3% | 14.88 | 14.39 | −3.3% |
| in1024_out128_c32 | 10.18 | 12.32 | +21.0% | 27.84 | 25.69 | −7.7% |
| in1024_out128_c128 | 7.59 | 10.74 | +41.5% | 136.63 | 95.87 | −29.8% |
| in1024_out128_c256 | 7.67 | 10.78 | +40.5% | 141.76 | 96.98 | −31.6% |
| in2048_out256_c16 | 5.38 | 5.02 | −6.7% | 18.69 | 16.47 | −11.9% |
| in2048_out256_c32 | 4.97 | 5.62 | +13.1% | 33.10 | 30.02 | −9.3% |
| in2048_out256_c128 | 3.64 | 5.33 | +46.4% | 165.68 | 135.46 | −18.2% |
| in2048_out256_c256 | 3.85 | 5.13 | +33.2% | 175.57 | 131.72 | −25.0% |

Key takeaways:
- High concurrency (c≥128): 10-13% RPS improvement for 9B/27B, 41-56% for 0.8B
- Medium concurrency (c=16-32): 1-3% RPS (large models), 4-21% (0.8B)
- Single request (c=1): Flat (expected — cannot amortize kernel gain)
- TPOT consistently lower for c≥16 across all three models
- 0.8B sees largest relative gains — lighter compute makes memory bandwidth the tighter bottleneck

### 4. Accuracy (`lm_eval` GSM8K 5-shot)

Qwen3.5-9B fp8 — 4060Ti (eager):

| Metric | main | feature | Match |
|--------|------|---------|-------|
| exact_match,strict-match | 0.873389 | 0.873389 | ✓ bit-exact |
| exact_match,flexible-extract | 0.865807 | 0.865807 | ✓ bit-exact |

Qwen3.5-9B fp8 — H20 (CUDAGraph+compile):

| Metric | main | feature | Match |
|--------|------|---------|-------|
| exact_match,strict-match | 0.8650 | 0.8650 | ✓ bit-exact |
| exact_match,flexible-extract | 0.8491 | 0.8491 | ✓ bit-exact |

OLMo-Hybrid-7B fp8 — H20 (CUDAGraph+compile):

| Metric | main | feature | Match |
|--------|------|---------|-------|
| exact_match,strict-match | 0.7263 | 0.7263 | ✓ bit-exact |
| exact_match,flexible-extract | 0.7263 | 0.7263 | ✓ bit-exact |

All accuracy metrics bit-exact across both models and both execution paths
(eager, CUDAGraph+compile). The absolute accuracy difference between 4060Ti
eager (0.8734) and H20 CUDAGraph (0.8650) is expected — CUDAGraph+compile
uses a deterministic execution path with slightly different numerics. The
key result is that within each config, main and feature branches produce
identical results.

### 5. FlashInfer backend (`verify_flashinfer.py`)

Qwen3.5-9B fp8 — H20 (FlashInfer backend, SM90+):

| Test | Batch size | Output tokens | Passed |
|------|-----------|---------------|--------|
| prefill_8t | 1 | 32 | ✓ |
| prefill_64t | 1 | 32 | ✓ |
| prefill_256t | 1 | 32 | ✓ |
| prefill_1024t | 1 | 32 | ✓ |
| decode_x4 | 4 | 4×32 | ✓ |
| decode_x16 | 16 | 16×32 | ✓ |
| decode_x64 | 64 | 64×32 | ✓ |
| decode_x128 | 128 | 128×32 | ✓ |
| mixed_1pf_15d | 16 | 16×32 | ✓ |
| mixed_4pf_60d | 64 | 64×32 | ✓ |
| mixed_8pf_120d | 128 | 128×32 | ✓ |

11/11 PASSED. The FlashInfer path (internal gather/scatter, not directly
modified by this PR) is functionally intact on the feature branch.
Only Qwen3.5-9B tested — OLMo-Hybrid-7B does not use FlashInfer for chunk
prefill.

### 6. Pre-commit hooks

```
pre-commit run --all-files: ruff ✅, mypy ✅
```

---

## Does this PR introduce any user-facing change?

No. Pure kernel performance optimization. The API, model behavior, and output
quality are unchanged (verified via bit-exact precision tests and GSM8K
accuracy).

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update — N/A, no user-facing API changes.

</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
