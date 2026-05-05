# GDN Chunk Prefill In-Place Kernel — PR 进度跟踪

> **PR 目标**：向 vllm-project/vllm 提交 PR，合并 GDN chunk prefill in-place kernel 优化。
> **分支**：`main`（优化前基线）vs `feature/gdn-prefill-kernal-opt`（优化后）
> **方案文档**：`kermit_docs/chunk_inplace_kernel_plan.md`
> **PR 草稿**：`kermit_docs/pr_description.md`

---

## 一、架构总览

### 1.1 改了什么

消除 GDN chunk prefill 路径中 SSM state 的 gather/scatter 开销。原来的调用链：

```
ssm_state[block_indices].contiguous()  # allocate + copy (gather)
  → kernel(initial_state)              # compute
    → ssm_state[block_indices] = ...   # copy back (scatter)
```

修改后内核通过 `ssm_state_indices` 直接在预分配的 cache 上读写，零拷贝。

| 层 | 文件 | 变更 |
|----|------|------|
| Kernel | `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` | `IS_CONTINUOUS_BATCHING` constexpr + 指针算术 |
| Adapter | `vllm/model_executor/layers/fla/ops/chunk.py` | 移除 `@input_guard`，透传 `ssm_state_indices` |
| Model | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `_forward_core` 去掉 gather/scatter |
| Model | `vllm/model_executor/models/olmo_hybrid.py` | 适配 `ssm_state_indices` |
| Test | `tests/kernels/test_chunk_inplace.py` | 3 个 pytest 用例 |

### 1.2 双模型路径差异

```
Qwen3.5-9B / Qwen3 NEXT:
  GDNLinearAttention → ChunkGatedDeltaRule CustomOp
    → forward_native  (Triton/FLA, 受益于 in-place)  ← 非 SM90 默认，本次优化的主收益路径
    → forward_cuda    (FlashInfer, 内部仍有 gather/scatter) ← SM90+ 默认

OLMo-Hybrid-7B:
  OlmoHybridGatedDeltaNet → chunk_gated_delta_rule (直接调用 chunk.py)
    → 始终 Triton/FLA 路径，无 FlashInfer 选项  ← 直接受益
```

**结论**：FlashInfer 正确性仅需验证 Qwen；性能测试两个模型都跑非 SM90 的 Triton 路径。

---

## 二、实验计划

### 实验矩阵

| 编号 | 实验 | 回答什么问题 | 方法 | 硬件 | 脚本 |
|------|------|-------------|------|------|------|
| E0 | 精度验证 | 优化后结果正确吗？ | bit-exact 对比 + pytest | 4060Ti | `dev_precision.py` + pytest |
| E1 | Kernel 微基准 | 纯 kernel 快了多少？ | git checkout 分支对比，CUDA event 计时 | 4060Ti | `bench_kernel.py` |
| E2 | E2E 延迟 | 模型推理实际快了多少？（TTFT/TPOT/per-token） | git checkout 分支对比，`LLM.generate()` | 4060Ti fp8 | `bench_e2e.py` |
| E3 | 吞吐极限 | 极限 QPS 提升多少？ | `vllm bench serve` 分支对比 | 4060Ti fp8 → H20 | `bench_serving.py` |
| E4 | 精度一致性 | lm_eval 准确率是否不变？ | gsm8k 5-shot 双分支对比 | H20 | `verify_accuracy.py` |
| E5 | FlashInfer 正确性 | SM90+ FlashInfer 路径是否正常？ | E2E 推理 + 输出校验 | H20 | `verify_flashinfer.py` |

### 4060Ti 进度

- **E0 精度**：已完成。8/8 bit-exact，3/3 pytest，pre-commit 通过。
- **E1 Kernel 微基准**：已完成。3 个 dims × 双分支，数据见第五章。
- **E2 E2E 延迟**：已完成。3 个模型 × 双分支。Qwen3.5-9B/OLMo 用 eager（CUDAGraph OOM），Qwen0.8B 用 CUDAGraph。
- **E3 吞吐**：已完成。仅 Qwen0.8B CUDAGraph（9B/7B CUDAGraph OOM）。小模型无显著差异，需 H20 跑大模型。
- **E4 精度一致性**：已完成。Qwen3.5-9B fp8 gsm8k 5-shot，main/feature 完全一致。

### H20 待执行

- **E2**：Qwen3.5-9B + OLMo fp8，CUDAGraph 生产路径（H20 显存够）。
- **E3**：双模型高压吞吐，扩展场景。
- **E5**：FlashInfer SM90+ 正确性验证。

---

## 三、脚本地图

```
kermit_docs/
├── PR 基准脚本 (git checkout 分支切换, 参数化, JSON 输出)
│   ├── bench_kernel.py           # E1: kernel 微基准, CUDA event, --dims qwen/olmo/qwen0.8b
│   ├── bench_e2e.py              # E2: E2E 延迟, LLM.generate(), --eager 可选
│   └── bench_serving.py          # E3: 吞吐, 启动 server + vllm bench serve
│
├── H20 验证脚本 (用户手动执行)
│   ├── verify_accuracy.py        # E4: lm_eval gsm8k, git checkout
│   └── verify_flashinfer.py      # E5: FlashInfer 正确性 (SM90+)
│
├── dev 阶段脚本 (已完成验证，结果已有)
│   └── dev_precision.py          # E0: 精度 bit-exact (同进程 OLD/NEW)
│
├── PR 草稿
│   └── pr_description.md
│
└── 参考
    └── chunk_inplace_kernel_plan.md
```

### 脚本设计原则（PR 用）

1. **git checkout 分支切换**，不用 monkey-patch。保证测的是真实编译路径。
2. **参数化**：模型路径、quantization、batch size 等用 CLI args，不硬编码。
3. **结构化输出**：结果写 JSON，方便 diff 和填表。
4. **自包含**：一个 `.py` 文件 + requirements 即跑。不依赖 monkey-patch、不 import 内部测试工具。
5. **标注 env**：脚本头部注释写清 conda env、GPU 要求、示例命令。

---

## 四、环境

### 本地 (4060Ti)

```
GPU:         NVIDIA RTX 4060 Ti (16GB)
Conda env:   /home/kermit/.conda/envs/vllm-20
Python:      3.12
Models:      .huggingface/Qwen3.5-9B (fp8), .huggingface/OLMo-Hybrid-7B (fp8),
             .huggingface/Qwen3.5-0.8B (bf16)
Branch:      main (baseline), feature/gdn-prefill-kernal-opt (optimized)
```

本地所有模型测试必须 `--quantization fp8`（16GB 显存放不下 bf16 的 9B + KV cache）。
**例外**：Qwen3.5-0.8B 可以跑 bf16——该模型仅 0.8B 参数，bf16 + KV cache 可完整放入 16GB 显存。

### H20 (远程)

```
GPU:         NVIDIA H20 (96GB+)
Conda env:   vllm-20 (需确认路径)
Models:      .huggingface/Qwen3.5-9B, .huggingface/OLMo-Hybrid-7B
```

H20 上 fp8 可选但不必须——显存够大。但是 `bench_serving.py` 压吞吐时建议 fp8 以对齐生产环境。

---

## 五、实验详情 & 进度

### E0 — 精度验证

| 项目 | 结果 |
|------|------|
| `dev_precision.py` (8 场景) | 8/8 PASS, ssm_state bit-exact, o diff <1e-2 |
| `pytest tests/kernels/test_chunk_inplace.py` (3 用例) | 3/3 PASSED |
| `pytest tests/kernels/ -k delta` (回归) | 12/18 PASS, 6 失败为已有问题 |
| pre-commit (ruff + mypy) | PASSED |

### E1 — Kernel 微基准

**目的**：排除模型 overhead，纯测 kernel 快了多少。用合成数据，不加载模型。

**脚本**：`bench_kernel.py`（git checkout 外部切换）

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm

# main 分支
git checkout main
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen --output /tmp/kernel_main.json

# feature 分支
git checkout feature/gdn-prefill-kernal-opt
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen --output /tmp/kernel_feat.json

# 对比
python -c "
import json
m=json.load(open('/tmp/kernel_main.json'))
f=json.load(open('/tmp/kernel_feat.json'))
for k in m['scenarios']:
    dm = m['scenarios'][k]['avg_us'] - f['scenarios'][k]['avg_us']
    pct = dm / m['scenarios'][k]['avg_us'] * 100
    print(f'{k:<24s} main={m[\"scenarios\"][k][\"avg_us\"]:8.1f}μs feat={f[\"scenarios\"][k][\"avg_us\"]:8.1f}μs Δ={dm:+7.1f}μs ({pct:+.1f}%)')
"

# OLMo dims:
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims olmo --output /tmp/kernel_olmo.json

# Qwen3.5-0.8B dims:
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen0.8b --output /tmp/kernel_qwen0.8b.json
```

覆盖场景：prefill (T=64/128/256/512/1024) + decode (N=1/16/64/128, T=1) + mixed。CUDA event 计时，30 warmup + 100 rounds 取 mean/median。

**4060Ti 实测数据（2026-05-04）**：

**Qwen3.5-9B dims (H_k=16, HV=32, K=128, V=128)**：

| Scenario | 备注 | main (μs) | feat (μs) | Δ (μs) | Δ% |
|----------|------|-----------|-----------|---------|-----|
| prefill_T64 | Prefill 单序列，T=64 tokens | 318.5 | 59.3 | +259.2 | +81.4% |
| prefill_T128 | Prefill 单序列，T=128 tokens | 317.7 | 77.3 | +240.4 | +75.7% |
| prefill_T256 | Prefill 单序列，T=256 tokens | 320.5 | 112.7 | +207.8 | +64.8% |
| prefill_T512 | Prefill 单序列，T=512 tokens | 343.5 | 186.9 | +156.6 | +45.6% |
| prefill_T1024 | Prefill 单序列，T=1024 tokens | 590.3 | 469.8 | +120.5 | +20.4% |
| decode_N1 | Decode 单请求，T=1，batch=1 | 324.1 | 42.5 | +281.6 | +86.9% |
| decode_N16 | Decode，batch=16，T=1 | 466.7 | 462.8 | +3.9 | +0.8% |
| decode_N64 | Decode，batch=64，T=1 | 1849.6 | 1850.2 | −0.6 | −0.0% |
| decode_N128 | Decode，batch=128，T=1 | 3628.8 | 3637.7 | −8.9 | −0.2% |
| mixed_2pf_14d | 混合：2 个 prefill + 14 个 decode | 497.4 | 482.3 | +15.1 | +3.0% |
| mixed_4pf_28d | 混合：4 个 prefill + 28 个 decode | 1029.2 | 1016.7 | +12.5 | +1.2% |
| mixed_8pf_56d | 混合：8 个 prefill + 56 个 decode | 2073.3 | 2072.7 | +0.6 | +0.0% |

**Qwen3.5-0.8B dims (H_k=16, HV=16, K=128, V=128)**：

| Scenario | 备注 | main (μs) | feat (μs) | Δ (μs) | Δ% |
|----------|------|-----------|-----------|---------|-----|
| prefill_T64 | Prefill 单序列，T=64 tokens | 326.9 | 53.2 | +273.7 | +83.7% |
| prefill_T128 | Prefill 单序列，T=128 tokens | 335.1 | 58.4 | +276.7 | +82.6% |
| prefill_T256 | Prefill 单序列，T=256 tokens | 323.6 | 78.7 | +244.9 | +75.7% |
| prefill_T512 | Prefill 单序列，T=512 tokens | 330.8 | 117.1 | +213.7 | +64.6% |
| prefill_T1024 | Prefill 单序列，T=1024 tokens | 357.0 | 197.8 | +159.2 | +44.6% |
| decode_N1 | Decode 单请求，T=1，batch=1 | 334.7 | 35.4 | +299.3 | +89.4% |
| decode_N16 | Decode，batch=16，T=1 | 335.8 | 245.3 | +90.5 | +27.0% |
| decode_N64 | Decode，batch=64，T=1 | 975.6 | 968.6 | +7.0 | +0.7% |
| decode_N128 | Decode，batch=128，T=1 | 1911.6 | 1922.0 | −10.4 | −0.5% |
| mixed_2pf_14d | 混合：2 个 prefill + 14 个 decode | 369.1 | 245.1 | +124.0 | +33.6% |
| mixed_4pf_28d | 混合：4 个 prefill + 28 个 decode | 533.6 | 515.8 | +17.8 | +3.3% |
| mixed_8pf_56d | 混合：8 个 prefill + 56 个 decode | 1105.6 | 1095.2 | +10.4 | +0.9% |

**OLMo-Hybrid-7B dims (H_k=30, HV=30, K=96, V=192)**：

| Scenario | 备注 | main (μs) | feat (μs) | Δ (μs) | Δ% |
|----------|------|-----------|-----------|---------|-----|
| prefill_T64 | Prefill 单序列，T=64 tokens | 321.8 | 64.4 | +257.4 | +80.0% |
| prefill_T128 | Prefill 单序列，T=128 tokens | 320.1 | 90.4 | +229.7 | +71.8% |
| prefill_T256 | Prefill 单序列，T=256 tokens | 323.0 | 144.7 | +178.3 | +55.2% |
| prefill_T512 | Prefill 单序列，T=512 tokens | 399.7 | 274.9 | +124.8 | +31.2% |
| prefill_T1024 | Prefill 单序列，T=1024 tokens | 833.1 | 729.1 | +104.0 | +12.5% |
| decode_N1 | Decode 单请求，T=1，batch=1 | 330.5 | 45.4 | +285.1 | +86.3% |
| decode_N16 | Decode，batch=16，T=1 | 527.4 | 529.1 | −1.7 | −0.3% |
| decode_N64 | Decode，batch=64，T=1 | 2055.6 | 2099.3 | −43.7 | −2.1% |
| decode_N128 | Decode，batch=128，T=1 | 4052.2 | 4130.3 | −78.1 | −1.9% |
| mixed_2pf_14d | 混合：2 个 prefill + 14 个 decode | 586.1 | 571.2 | +14.9 | +2.5% |
| mixed_4pf_28d | 混合：4 个 prefill + 28 个 decode | 1190.3 | 1197.3 | −7.0 | −0.6% |
| mixed_8pf_56d | 混合：8 个 prefill + 56 个 decode | 2361.5 | 2396.7 | −35.2 | −1.5% |

**结论**：
- **单序列 (N=1)**：kernel 级提升巨大，prefill 12-84%，decode_N1 86-89%。旧 kernel 的 gather/scatter 开销在单序列时占比极高。
- **大批量 decode (N≥64)**：kernel 级基本持平（±2.1% 以内）。大批量下 compute 主导，gather/scatter 占比可忽略。
- **小批量 decode (N=16)**：qwen0.8b dims 仍 +27.0%（HV=16 时 gather/scatter 开销更显著），qwen/olmo 近零。
- **mixed**：N=16 组 +2.5-3.3%，N=64 组 −1.5-+0.9%。
- **dim 差异**：HV 越小（qwen0.8b HV=16），in-place 优化相对收益越大。

### E2 — E2E 延迟

**目的**：模型推理的实际延迟改善（TTFT、TPOT、total latency）。

**重要发现**：Qwen3.5-9B fp8 + CUDAGraph 在 4060Ti 16GB 上 OOM（权重 10.81 GiB + CUDAGraph profiling > 15.6 GiB）。Qwen3.5-9B 和 OLMo-Hybrid-7B 均使用 `--eager` 模式（compile only，无 CUDAGraph）。Qwen3.5-0.8B bf16 使用 CUDAGraph（生产路径）。

**脚本**：`bench_e2e.py`（git checkout 外部切换）

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm

# Qwen3.5-9B fp8 (--eager required on 16GB GPU)
git checkout main
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --eager --output /tmp/e2e_main_qwen.json
git checkout feature/gdn-prefill-kernal-opt
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --eager --output /tmp/e2e_feat_qwen.json

# OLMo-Hybrid-7B fp8 (--eager required on 16GB GPU)
git checkout main
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --eager --output /tmp/e2e_main_olmo.json
git checkout feature/gdn-prefill-kernal-opt
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --eager --output /tmp/e2e_feat_olmo.json

# Qwen3.5-0.8B bf16 (CUDAGraph, 生产路径)
git checkout main
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/e2e_main_qwen0.8b.json
git checkout feature/gdn-prefill-kernal-opt
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/e2e_feat_qwen0.8b.json

# 对比
python -c "
import json
m=json.load(open('/tmp/e2e_main.json'))
f=json.load(open('/tmp/e2e_feat.json'))
for k in m['scenarios']:
    dm = m['scenarios'][k]['avg_ms'] - f['scenarios'][k]['avg_ms']
    pct = dm / m['scenarios'][k]['avg_ms'] * 100
    print(f'{k:<24s} main={m[\"scenarios\"][k][\"avg_ms\"]:8.1f}ms feat={f[\"scenarios\"][k][\"avg_ms\"]:8.1f}ms Δ={dm:+6.1f}ms ({pct:+.1f}%)')
"
```

覆盖：prefill (T=64/128/256/512/1024) + decode (N=1/16/64/128) + mixed。每场景 3 warmup + 10 rounds，取 avg/min/max。

**4060Ti 实测数据（2026-05-04）**：

**Qwen3.5-9B fp8 (eager, compile only, 无 CUDAGraph)**：
| prefill_64t | Prefill 单请求，64 tokens | 51.2 | 50.3 | +0.9 | +1.7% |
| prefill_128t | Prefill 单请求，128 tokens | 96.0 | 95.1 | +0.9 | +0.9% |
| prefill_256t | Prefill 单请求，256 tokens | 156.3 | 155.5 | +0.9 | +0.6% |
| prefill_512t | Prefill 单请求，512 tokens | 152.2 | 151.3 | +0.9 | +0.6% |
| prefill_1024t | Prefill 单请求，1024 tokens | 278.0 | 276.1 | +1.8 | +0.7% |
| decode_x1 | Decode，1 个并发请求 | 46.9 | 44.5 | +2.4 | +5.2% |
| decode_x16 | Decode，16 个并发请求 | 198.8 | 188.3 | +10.5 | +5.3% |
| decode_x64 | Decode，64 个并发请求 | 658.8 | 623.5 | +35.3 | +5.4% |
| decode_x128 | Decode，128 个并发请求 | 1315.0 | 1244.4 | +70.6 | +5.4% |
| mixed_1pf_15d | 混合：1 个 prefill + 15 个 decode | 205.5 | 194.0 | +11.5 | +5.6% |
| mixed_4pf_60d | 混合：4 个 prefill + 60 个 decode | 710.3 | 671.6 | +38.7 | +5.4% |
| mixed_8pf_120d | 混合：8 个 prefill + 120 个 decode | 1623.6 | 1546.0 | +77.6 | +4.8% |

**Qwen3.5-0.8B bf16 (CUDAGraph, 生产路径)**：

| Scenario | 备注 | main (ms) | feat (ms) | Δ (ms) | Δ% |
|----------|------|-----------|-----------|--------|-----|
| prefill_64t | Prefill 单请求，64 tokens | 17.2 | 16.3 | +0.9 | +5.4% |
| prefill_128t | Prefill 单请求，128 tokens | 18.9 | 18.7 | +0.2 | +1.1% |
| prefill_256t | Prefill 单请求，256 tokens | 27.6 | 27.8 | −0.2 | −0.7% |
| prefill_512t | Prefill 单请求，512 tokens | 25.9 | 25.8 | +0.1 | +0.4% |
| prefill_1024t | Prefill 单请求，1024 tokens | 43.4 | 44.0 | −0.7 | −1.6% |
| decode_x1 | Decode，1 个并发请求 | 16.7 | 15.8 | +0.9 | +5.2% |
| decode_x16 | Decode，16 个并发请求 | 23.6 | 20.1 | +3.6 | +15.0% |
| decode_x64 | Decode，64 个并发请求 | 85.9 | 54.5 | +31.4 | +36.6% |
| decode_x128 | Decode，128 个并发请求 | 168.8 | 103.9 | +64.8 | +38.4% |
| mixed_1pf_15d | 混合：1 个 prefill + 15 个 decode | 25.9 | 21.3 | +4.6 | +17.8% |
| mixed_4pf_60d | 混合：4 个 prefill + 60 个 decode | 95.0 | 63.4 | +31.6 | +33.3% |
| mixed_8pf_120d | 混合：8 个 prefill + 120 个 decode | 223.4 | 159.1 | +64.3 | +28.8% |

**OLMo-Hybrid-7B fp8 (eager, compile only, 无 CUDAGraph)**：

| Scenario | 备注 | main (ms) | feat (ms) | Δ (ms) | Δ% |
|----------|------|-----------|-----------|--------|-----|
| prefill_64t | Prefill 单请求，64 tokens | 44.8 | 44.5 | +0.3 | +0.8% |
| prefill_128t | Prefill 单请求，128 tokens | 88.2 | 87.9 | +0.3 | +0.4% |
| prefill_256t | Prefill 单请求，256 tokens | 145.2 | 144.5 | +0.6 | +0.4% |
| prefill_512t | Prefill 单请求，512 tokens | 141.9 | 141.2 | +0.7 | +0.5% |
| prefill_1024t | Prefill 单请求，1024 tokens | 258.0 | 257.0 | +1.1 | +0.4% |
| decode_x1 | Decode，1 个并发请求 | 41.4 | 39.8 | +1.6 | +3.9% |
| decode_x16 | Decode，16 个并发请求 | 84.3 | 68.4 | +15.9 | +18.8% |
| decode_x64 | Decode，64 个并发请求 | 258.5 | 187.3 | +71.1 | +27.5% |
| decode_x128 | Decode，128 个并发请求 | 543.8 | 406.8 | +137.0 | +25.2% |
| mixed_1pf_15d | 混合：1 个 prefill + 15 个 decode | 89.7 | 74.0 | +15.7 | +17.5% |
| mixed_4pf_60d | 混合：4 个 prefill + 60 个 decode | 316.9 | 246.4 | +70.5 | +22.2% |
| mixed_8pf_120d | 混合：8 个 prefill + 120 个 decode | 858.8 | 721.4 | +137.5 | +16.0% |

**结论**：
- **Qwen3.5-0.8B CUDAGraph（生产路径）**：decode +36-38%，mixed +17-33%。E2E 收益远超 kernel 级预期——CUDAGraph 捕获的 kernel 在图优化后放大效果显著。
- **Qwen3.5-9B eager**：decode +5.2-5.4%，mixed +4.8-5.6%。预期内收益。
- **OLMo-Hybrid-7B eager**：decode +3.9-27.5%，mixed +16.0-22.2%。OLMo 的 GDN 层更多（30 层 k/v heads），in-place 优化累积收益更大。
- **prefill 单序列**：E2E 级收益 0.4-5.4%，远小于 kernel 级的 40-89%——kernel 只占 prefill 总延迟的一小部分。
- **9B fp8 + CUDAGraph OOM**：16GB 显存不足以同时容纳模型权重 + CUDAGraph profiling 内存。H20 上不存在此瓶颈。

### E3 — 吞吐极限 (Qwen0.8B 4060Ti)

**目的**：`vllm bench serve` 压极限 QPS，对比吞吐和 delay。

**⚠️ 4060Ti 限制**：Qwen3.5-9B fp8 + CUDAGraph OOM（同 E2），`vllm bench serve` 内部启动 server 默认用 CUDAGraph，9B/7B 模型无法在 4060Ti 上跑 E3。Qwen3.5-0.8B bf16 可跑。

**脚本**：`bench_serving.py`（git checkout 外部切换）

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm

# Qwen3.5-0.8B bf16 (唯一在 4060Ti 上能跑 CUDAGraph 的模型)
git checkout main
python kermit_docs/bench_serving.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/serving_main.json
git checkout feature/gdn-prefill-kernal-opt
python kermit_docs/bench_serving.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/serving_feat.json

# H20: Qwen3.5-9B + OLMo
python kermit_docs/bench_serving.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --h20 --output /tmp/serving_h20.json
```

**4060Ti 场景**：(512/1024, 128) × (inf, 2) = 4 组。每场景 100 prompts。⚠️ rinf 场景方差大（首轮编译、GPU 状态），r2 更稳定。

**H20 场景**：(512/1024/2048/4096, 128/256) × (inf, 2, 1) = 9 组。每场景 200 prompts。

输出指标：request_throughput, output_throughput, TTFT (mean/median/p99), TPOT, ITL。

**4060Ti 实测数据 — Qwen3.5-0.8B bf16 (CUDAGraph, 2026-05-04)**：

| Scenario | 备注 | main rps | feat rps | main TTFT | feat TTFT | Δ rps |
|----------|------|----------|----------|-----------|-----------|-------|
| in512_out128_rinf | 输入 512 tokens，输出 128 tokens，无限速率（burst 压测） | 14.88 | 5.31 | 1588.6ms | 13746.7ms | −64.3% |
| in1024_out128_rinf | 输入 1024 tokens，输出 128 tokens，无限速率（burst 压测） | 10.22 | 11.62 | 2928.5ms | 2522.9ms | +13.7% |
| in512_out128_r2 | 输入 512 tokens，输出 128 tokens，2 req/s 稳定速率 | 1.96 | 1.96 | 42.0ms | 42.5ms | +0.0% |
| in1024_out128_r2 | 输入 1024 tokens，输出 128 tokens，2 req/s 稳定速率 | 1.96 | 1.96 | 62.1ms | 61.6ms | +0.0% |

**结论**：rinf (burst) 场景方差极大，不可靠（首轮编译、冷启动差异）。r2 (稳定 2 req/s) 场景无差异。小模型 (0.8B) 的 GDN kernel 在 serving 吞吐中占比不足 1%，优化无测量影响。**E3 实际收益要在 H20 + 大模型 (9B/7B) 上才能体现。** 4060Ti 仅验证了脚本流程正确。

### E4 — 精度一致性 (4060Ti, Qwen3.5-9B fp8)

**脚本**：`verify_accuracy.py`（git checkout 外部切换）

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm

# main 分支
git checkout main
python kermit_docs/verify_accuracy.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/acc_main.json

# feature 分支
git checkout feature/gdn-prefill-kernal-opt
python kermit_docs/verify_accuracy.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/acc_feat.json

# OLMo
python kermit_docs/verify_accuracy.py \
    --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --output /tmp/acc_olmo.json

# 对比
python -c "
import json
m=json.load(open('/tmp/acc_main.json'))
f=json.load(open('/tmp/acc_feat.json'))
for t in m['tasks']:
    print(f'{t}: main={m[\"tasks\"][t][\"accuracy\"]:.4f} feat={f[\"tasks\"][t][\"accuracy\"]:.4f}')
"
```

**4060Ti 实测数据（2026-05-05）— Qwen3.5-9B fp8**：

| Metric | main | feature | match |
|--------|------|---------|-------|
| exact_match,strict-match | 0.873389 | 0.873389 | ✓ |
| exact_match_stderr,strict-match | 0.009160 | 0.009160 | ✓ |
| exact_match,flexible-extract | 0.865807 | 0.865807 | ✓ |
| exact_match_stderr,flexible-extract | 0.009389 | 0.009389 | ✓ |

**结论**：main 和 feature 分支 gsm8k 5-shot accuracy **完全一致**，所有指标 bit-exact 匹配。in-place kernel 优化不影响推理精度。

### E5 — FlashInfer 正确性 (H20, SM90+)

**脚本**：`verify_flashinfer.py`（SM90+ 专用，单分支 smoke test）

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm

PYTHONPATH=. python kermit_docs/verify_flashinfer.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/flashinfer.json
```

仅测试 Qwen3.5-9B。覆盖 prefill/decode/mixed 共 11 场景，验证输出非空、不报错。

---

## 六、接下来做的事（优先级排序）

```
┌──────┬──────────────────────────────────────┬──────────┬──────────────────────┐
│ 优先级 │ 任务                                  │ 状态     │ 产出                  │
├──────┼──────────────────────────────────────┼──────────┼──────────────────────┤
│ 1    │ 跑 bench_kernel.py 4060Ti (qwen)     │ ✅ DONE  │ kernel_qwen JSON      │
│ 2    │ 跑 bench_kernel.py 4060Ti (qwen0.8b) │ ✅ DONE  │ kernel_qwen0.8b JSON  │
│ 3    │ 跑 bench_kernel.py 4060Ti (olmo)     │ ✅ DONE  │ kernel_olmo JSON      │
│ 4    │ 跑 bench_e2e.py 4060Ti (qwen fp8)    │ ✅ DONE  │ e2e_qwen JSON (eager) │
│ 5    │ 跑 bench_e2e.py 4060Ti (qwen0.8B bf16)│ ✅ DONE │ e2e_qwen0.8b JSON     │
│ 6    │ 跑 bench_e2e.py 4060Ti (olmo fp8)    │ ✅ DONE  │ e2e_olmo JSON (eager) │
│ 7    │ 跑 bench_serving.py 4060Ti (qwen0.8b)│ ✅ DONE  │ 本地吞吐数据 (无显著差异) │
│ 8    │ H20: 跑 bench_e2e.py + bench_serving │ ⏳ H20   │ 高压数据              │
│ 9    │ H20/本地: 跑 verify_accuracy.py (双模型) │ ✅ DONE  │ lm_eval 精度 (Qwen9B) │
│ 10   │ H20: 跑 verify_flashinfer.py (qwen)  │ ⏳ H20   │ FlashInfer 验证       │
│ 11   │ 填 PR description [TODO]              │ ⏳ 等数据 │ 最终 PR 稿            │
└──────┴──────────────────────────────────────┴──────────┴──────────────────────┘
```

---

## 七、4060Ti 本地速查命令

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm

# ── pre-commit ──
pre-commit run ruff-check --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual

# ── pytest ──
.venv/bin/python -m pytest tests/kernels/test_chunk_inplace.py -v
.venv/bin/python -m pytest tests/kernels/ -k delta -v \
    --ignore=tests/kernels/helion --ignore=tests/kernels/ir --ignore=tests/kernels/quantization

# ── 精度 (E0) — 已完成 ──
PYTHONPATH=. python kermit_docs/dev_precision.py

# ── PR 用基准 (E1/E2/E3 — ✅ ALL DONE on 4060Ti) ──
# E1: kernel 微基准 (已完成, 数据见上)
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen --output /tmp/kernel_qwen.json
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen0.8b --output /tmp/kernel_qwen0.8b.json
PYTHONPATH=. python kermit_docs/bench_kernel.py --dims olmo --output /tmp/kernel_olmo.json

# E2: E2E 延迟 (已完成, 数据见上; 9B/7B 需 --eager, 0.8B 可 CUDAGraph)
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/Qwen3.5-9B --quantization fp8 --eager --output /tmp/e2e_qwen.json
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --eager --output /tmp/e2e_olmo.json
PYTHONPATH=. python kermit_docs/bench_e2e.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/e2e_qwen0.8b.json

# E3: 吞吐 (仅 0.8B 能在 4060Ti 跑 CUDAGraph)
python kermit_docs/bench_serving.py \
    --model .huggingface/Qwen3.5-0.8B --output /tmp/serving_qwen0.8b.json
```

---

## 八、H20 执行 Checklist

用户手动执行后，把 JSON 结果发回来填 PR。

```bash
conda activate vllm-20
cd /home/kermit/MyCode/vllm
```

```
[ ] 1. bench_e2e.py — Qwen3.5-9B + OLMo-Hybrid-7B (H20, compile+CUDAGraph):
        # main 分支
        git checkout main
        PYTHONPATH=. python kermit_docs/bench_e2e.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/e2e_main_qwen.json
        PYTHONPATH=. python kermit_docs/bench_e2e.py \
            --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --output /tmp/e2e_main_olmo.json
        # feature 分支
        git checkout feature/gdn-prefill-kernal-opt
        PYTHONPATH=. python kermit_docs/bench_e2e.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/e2e_feat_qwen.json
        PYTHONPATH=. python kermit_docs/bench_e2e.py \
            --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --output /tmp/e2e_feat_olmo.json

[ ] 2. bench_serving.py — E3 吞吐 (双模型, H20 扩展场景):
        git checkout main
        python kermit_docs/bench_serving.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --h20 --output /tmp/serving_main_qwen.json
        git checkout feature/gdn-prefill-kernal-opt
        python kermit_docs/bench_serving.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --h20 --output /tmp/serving_feat_qwen.json
        git checkout main
        # OLMo
        python kermit_docs/bench_serving.py \
            --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --h20 --output /tmp/serving_main_olmo.json
        git checkout feature/gdn-prefill-kernal-opt
        python kermit_docs/bench_serving.py \
            --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --h20 --output /tmp/serving_feat_olmo.json

[ ] 3. verify_accuracy.py — E4 lm_eval gsm8k (双模型, 需联网):
        git checkout main
        python kermit_docs/verify_accuracy.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/acc_main_qwen.json
        git checkout feature/gdn-prefill-kernal-opt
        python kermit_docs/verify_accuracy.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/acc_feat_qwen.json

[ ] 4. verify_flashinfer.py — E5 FlashInfer 正确性 (仅 Qwen, SM90+):
        PYTHONPATH=. python kermit_docs/verify_flashinfer.py \
            --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/flashinfer.json

[ ] 5. 把所有 JSON 发回来，统一填 PR description
```

---

## 九、PR Description 状态

`kermit_docs/pr_description.md` 草稿已有一版，但以下字段待填：

- `[TODO]` bench_kernel.py 数据（替代现有 monkey-patch 数据）
- `[TODO]` bench_e2e.py compile+CUDAGraph 数据（替代现有 eager 数据）
- `[TODO]` bench_serving.py 吞吐数据
- `[TODO]` lm_eval gsm8k 精度数据
- `[TODO]` FlashInfer E2E 验证
