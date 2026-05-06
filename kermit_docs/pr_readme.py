# vLLM GatedDeltaNet Kernel 优化 PR 指南

> 本文档供 Claude Code 使用，用于根据实际代码改动自动填充 PR 的各个部分。
> 所有规范均来自 vLLM 官方贡献指南及真实已 merge / 真实 review 过程的 PR 分析。

-----

## 一、PR 标题规范

### 格式

```
[Kernel] Optimize GatedDeltaNet Triton kernel, Xx speedup for Qwen3.5/Qwen3 NEXT
```

### 规则说明

- **前缀必须用方括号**，vLLM 不用 `feat:` / `fix:` 风格，有自己的分类系统（来自官方 contributing guide）
- 核心类别是 `[Kernel]`，因为改动的是 Triton kernel 实现
- **speedup 倍数直接写进标题**，这是 vLLM kernel PR 的强烈惯例
  - 参考：PR #16173 标题 `[Kernel] support merge_attn_states CUDA kernel, 3x speedup`
  - 作用：帮 reviewer 在 PR 列表里直接判断优先级，不需要进去读 description
- 模型名 Qwen3.5 / Qwen3 NEXT 出现在标题里，方便按模型检索
- 如果改动同时触及多个类别，所有前缀都要列出，例如 `[Kernel][Model]` 或 `[Bugfix][Kernel]`
  - 参考：PR #22593 标题 `[Bugfix][Kernel] Support partial rotary embedding for MRoPE triton kernel`
- 标题在 review 过程中可以改，PR #14431 的标题在 review 期间改了 3 次——开 PR 时标题不需要完美，内容对才是关键

### 完整前缀参考表

|前缀          |适用场景                             |
|------------|---------------------------------|
|`[Kernel]`  |CUDA / Triton kernel 改动 ← **主前缀**|
|`[Model]`   |涉及具体模型，模型名需出现在标题                 |
|`[Core]`    |LLMEngine、Scheduler 等核心逻辑        |
|`[Frontend]`|OpenAI API server、LLM class      |
|`[Bugfix]`  |bug 修复                           |
|`[Perf]`    |性能优化（有时与 Kernel 并用）              |
|`[CI/Build]`|CI 或构建系统改动                       |
|`[Doc]`     |文档修复                             |
|`[Misc]`    |以上都不合适时，尽量少用                     |

-----

## 二、PR Description 完整模板

以下是供 Claude Code 填充的模板，`[TODO: ...]` 标注的部分需要根据实际改动自动填写。

-----

```markdown
**TLDR:** [TODO: 一句话总结核心结论 + 数字。
格式参考 PR #14431 的写法：
"This PR optimizes `fused_recurrent_gated_delta_rule` by <改动方法>,
achieving **Xx speedup** on <硬件> for Qwen3.5 / Qwen3 NEXT.
With these changes, the kernel is only X% slower than <对照 baseline>."

注意：TLDR 里就要有加粗的数字，不能只说"improves performance"。]

---

## What this PR does / why we need it

### Background

The GatedDeltaNet (GDN) linear attention kernel (`fused_recurrent_gated_delta_rule`) is a
core compute primitive for both Qwen3.5 and Qwen3 NEXT. These models use a hybrid
architecture that interleaves GDN layers with full attention at a 3:1 ratio, making GDN
decode performance the dominant per-token bottleneck at small batch sizes.

Further optimization of GatedDeltaNet kernels is explicitly on the vLLM roadmap
(see the [Qwen3 NEXT release blog](https://blog.vllm.ai/2025/09/11/qwen3-next.html):
"Further kernel optimizations for GatedDeltaNet layers").

### Root cause of the bottleneck

[TODO: 具体说明原始 kernel 慢在哪里。要有技术深度，不能只说"性能不好"。例如：
- decode 阶段 batch_size=1 时，原有 BLOCK_SIZE 配置导致 SM occupancy 低于 X%
- num_warps=4 在 H100 上未充分利用 warp-level parallelism
- 存在冗余的 global memory load（具体指哪个 tensor）
- shared memory bank conflict 在哪个 access pattern 出现
- 具体取决于你的改动，如实填写]

### Changes in this PR

[TODO: 列出关键改动，每条说清楚改了什么、为什么这样改。例如：
- Changed `BLOCK_SIZE_K` from X to Y: reduces register pressure while maintaining L2 cache reuse
- Increased `num_warps` from 4 to 8 on SM90 (H100): better hides memory latency at small batch
- Fused the gate computation into the main recurrent loop: eliminates one global memory round trip
- Added per-hardware `tl.autotune` configs for H100/A100/H20
- 如实填写]

---

## Affected models

Both Qwen3.5 and Qwen3 NEXT share the same underlying Triton kernel. The kernel file
and the two model files calling it:

| Layer | File | Call site |
|-------|------|-----------|
| Kernel | `vllm/attention/ops/[TODO: 具体文件名]` | `fused_recurrent_gated_delta_rule` |
| Qwen3.5 model | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5GatedDeltaNet.forward()` line [TODO] |
| Qwen3 NEXT model | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextGatedDeltaNet.forward()` line [TODO] |

**Scope of change:** This PR modifies the kernel implementation only. The model-level
calling code in `qwen3_5.py` / `qwen3_next.py` is [TODO: unchanged / modified as follows: ...]

---

## Performance results

### Test environment
```

Hardware:    [TODO: e.g., 8x NVIDIA H100 80GB SXM5]
Model:       [TODO: e.g., Qwen3.5-27B, bf16, TP=8]
vLLM:        [TODO: commit SHA]
Triton:      [TODO: version, e.g., 3.2.0]
PyTorch:     [TODO: version]
CUDA:        [TODO: version]

```
### Kernel-level microbenchmark

Isolated kernel timing (before vs. after), sweeping across representative input shapes.
Format follows PR #16173.

| tokens | heads | head_size | dtype   | device | before (ms) | after (ms) | speedup |
|--------|-------|-----------|---------|--------|-------------|------------|---------|
| [TODO] | [TODO]| [TODO]    | bf16    | [TODO] | [TODO]      | [TODO]     | [TODO]x |
| [TODO] | [TODO]| [TODO]    | bf16    | [TODO] | [TODO]      | [TODO]     | [TODO]x |
| [TODO] | [TODO]| [TODO]    | float16 | [TODO] | [TODO]      | [TODO]     | [TODO]x |
| ...    |       |           |         |        |             |            |         |

> The kernel shows best improvement at [TODO: 描述在什么条件下收益最大，例如 small batch / long sequence]
> and more modest gains at [TODO: 描述收益有限的情况，如 large batch].
> This is expected because [TODO: 简短技术解释，例如：at large batch the kernel becomes compute-bound
> rather than memory-bound, so the tiling optimization has less impact].

### End-to-end serving benchmark

Full serving benchmark using `benchmarks/benchmark_serving.py`.
Follows the format of PR #14431 — full output block, not just summary numbers.

Benchmark command:
```bash
[TODO: 贴出完整命令，例如：
python benchmarks/benchmark_serving.py \
    --model Qwen/Qwen3.5-27B \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --tensor-parallel-size 8]
```

**Before (main branch):**

```
============ Serving Benchmark Result ============
Successful requests:                     [TODO]
Benchmark duration (s):                  [TODO]
Total input tokens:                      [TODO]
Total generated tokens:                  [TODO]
Request throughput (req/s):              [TODO]
Output token throughput (tok/s):         [TODO]
Total Token throughput (tok/s):          [TODO]
---------------Time to First Token----------------
Mean TTFT (ms):                          [TODO]
Median TTFT (ms):                        [TODO]
P99 TTFT (ms):                           [TODO]
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          [TODO]
Median TPOT (ms):                        [TODO]
P99 TPOT (ms):                           [TODO]
---------------Inter-token Latency----------------
Mean ITL (ms):                           [TODO]
Median ITL (ms):                         [TODO]
P99 ITL (ms):                            [TODO]
==================================================
```

**After (this PR):**

```
============ Serving Benchmark Result ============
[TODO: 同格式贴 after 结果]
==================================================
```

-----

## Correctness verification

**This section must be present before requesting review.** Reviewers will explicitly ask
for it if missing — in PR #14431, a reviewer commented “please publish accuracy test as well”
mid-review, causing an unnecessary round-trip delay.

Kernel outputs are validated against a float32 reference implementation across multiple
dtypes and input shapes. Format follows PR #16173.

```
[TODO: 贴出 pytest 完整输出，例如：

$ pytest -s tests/kernels/test_gated_delta_net.py
------------------------------------------------------------
NUM_TOKENS:256, NUM_HEADS:16, HEAD_SIZE:128, DTYPE:torch.bfloat16, Device: NVIDIA H100
 Reference time: X.XXXms
   Kernel time: X.XXXms, Speedup: X.XXx
Output all match, max abs diff: X.XXe-XX
------------------------------------------------------------
NUM_TOKENS:1024, NUM_HEADS:16, HEAD_SIZE:128, DTYPE:torch.float16, Device: NVIDIA H100
 Reference time: X.XXXms
   Kernel time: X.XXXms, Speedup: X.XXx
Output all match, max abs diff: X.XXe-XX
------------------------------------------------------------
]
```

Key correctness metrics:

- max abs diff vs. float32 reference: `[TODO: X.XXe-XX]`
- atol / rtol: `[TODO: atol=X.Xe-X, rtol=X.Xe-X]`
- dtypes tested: bf16, fp16 [TODO: fp32 if applicable]

[TODO: 如果跑了 lm_eval 端到端 accuracy 验证，一并贴出，参考 PR #14152 的格式：

$ lm_eval –model vllm   
–model_args pretrained=Qwen/Qwen3.5-27B   
–tasks gsm8k –num_fewshot 5 –batch_size auto –limit 500

|Tasks|Version|Filter          |n-shot|Metric     |Value|Stderr |
|-----|-------|----------------|------|-----------|-----|-------|
|gsm8k|3      |flexible-extract|5     |exact_match|X.XXX|±X.XXXX|
|]    |       |                |      |           |     |       |

-----

## Does this PR introduce any user-facing change?

No. This is a pure kernel performance optimization. The API, model behavior, and output
quality are unchanged within numerical tolerance (verified above).

[TODO: 如果改了暴露给用户的参数（如 chunk size、block size 可配置化），改为 Yes 并描述具体变化]

-----

## How was this patch tested?

1. **Correctness:** `pytest tests/kernels/test_gated_delta_net.py` — output matches float32
   reference within atol=[TODO] / rtol=[TODO] across bf16 / fp16, multiple input shapes
1. **Performance:** `benchmarks/benchmark_serving.py` on [TODO: 硬件] with Qwen3.5-[TODO]
   (full results in Performance section above)
1. **Regression:** `pytest tests/kernels/ -k delta` — all existing tests pass
1. [TODO: 如有 lm_eval，写：**Accuracy:** gsm8k 5-shot on Qwen3.5-[TODO] — X.XXX (within expected variance of baseline)]

-----

## Upstream / attribution note

The `fused_recurrent_gated_delta_rule` kernel originates from the
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) project.
The original vLLM integration of this kernel for Qwen3 NEXT was done in close collaboration
with the Flash Linear Attention team (Yu Zhang et al.), who specifically reviewed the
kernel numerics.

This PR modifies vLLM’s local copy of the kernel.

[TODO: 选择以下之一填写：

- “These optimizations have been reviewed / discussed with the Flash Linear Attention team.”
- “These changes are specific to vLLM’s deployment configuration and do not require upstreaming.”
- “A corresponding contribution to fla-org/flash-linear-attention is planned / has been opened at <link>.”
  ]

-----

## Checklist

- [ ] PR title follows `[Kernel]` prefix convention with speedup in title
- [ ] `Signed-off-by` present in **every** commit (`git commit -s`)
- [ ] `pre-commit run --all-files` passes (ruff, isort, mypy)
- [ ] Kernel-level microbenchmark table included (before / after / speedup)
- [ ] End-to-end `benchmark_serving.py` full output included (before & after)
- [ ] Correctness test with `max abs diff` included
- [ ] ROCm / AMD path confirmed unbroken (or explicitly scoped to CUDA-only with justification)
- [ ] No user-facing API changes

```
---

## 三、Commit Message 规范

vLLM commit message **格式宽松**，没有强制 Conventional Commits，实际惯例是：
```

Optimize GatedDeltaNet Triton kernel tiling for Qwen3.5/Qwen3 NEXT

Signed-off-by: Your Name [your@email.com](mailto:your@email.com)

```
或者直接复用 PR 标题：
```

[Kernel] Optimize GatedDeltaNet Triton kernel, Xx speedup for Qwen3.5/Qwen3 NEXT

Signed-off-by: Your Name [your@email.com](mailto:your@email.com)

```
**`Signed-off-by` 是硬性要求（DCO）**，缺一个 commit 整个 PR 就被 DCO bot 标红。

- 提交时：`git commit -s -m "your message"`
- 忘加了补救：`git rebase HEAD~N --signoff`
- 只有最后一个 commit 忘了：`git commit --amend -s`

---

## 四、关键注意事项汇总

### ✅ 通用 kernel PR 必须做的

| 注意点 | 来源 PR | 具体说明 |
|--------|---------|---------|
| **TLDR 第一行，加粗写核心数字** | PR #14431 | PR #14431 的 description 第一行就是 "**TLDR:** This PR adds... **25% improvement in throughput** for llama3.1-8b on H100"。Reviewer 扫 PR 列表时第一眼看到数字，才会优先进来 review |
| **标题直接写 speedup 倍数** | PR #16173 | PR #16173 标题是 `[Kernel] support merge_attn_states CUDA kernel, 3x speedup`。这是 vLLM kernel PR 的强烈惯例，不是在炫耀，是帮 reviewer 在几十个 PR 里快速判断优先级 |
| **同时提供 kernel microbenchmark + E2E serving benchmark** | PR #14431, #16173 | PR #16173 同时给了精确的 kernel 表格（多 dtype / token size）和端到端的 E2E 数字。只给 kernel latency 会被质疑"端到端收益不清楚"；两层数据互相印证说服力最强 |
| **E2E benchmark 要贴 `benchmark_serving.py` 完整输出块** | PR #14431 | PR #14431 贴了完整的 `Serving Benchmark Result` 块（含 TTFT、TPOT、ITL、req/s），不是只贴两个数字。不完整的数据给 reviewer 质疑空间 |
| **correctness 部分必须给 `max abs diff` 具体数值** | PR #16173 | PR #16173 的 correctness 部分贴了完整 pytest 输出，含 `max abs diff: X.XXe-XX` 这样的具体数字。不能只写"测试通过"，reviewer 对数值精度非常敏感 |
| **提前在 PR 里贴好 accuracy test，不要等 reviewer 要** | PR #14431 review 过程 | PR #14431 review 过程中 reviewer 直接评论 "please publish accuracy test as well"，贡献者才补上。这种来回拖进度，提前贴好能省掉一轮 review cycle |
| **收益有条件性时，主动说明适用范围和原因** | PR #22593 (top-k sampling kernel) | PR #22593 明确说明 "The kernel works better with larger batch sizes and K < vocab_size * 0.03"，并给出了技术原因。如果只报峰值 speedup 而不说明条件，reviewer 自己用其他配置测出来和你不一样时会直接质疑 |

### ⚠️ GDN kernel 特有注意点

| 注意点 | 来源 | 具体说明 |
|--------|------|---------|
| **kernel 有上游归属，PR 里需 acknowledge** | vLLM blog (Qwen3 NEXT 发布) | blog 中明确写道："Flash Linear Attention team, including Yu Zhang, etc. for reviewing the gated deltanet attention kernels and improving the numerics." `fused_recurrent_gated_delta_rule` 来自 `fla-org/flash-linear-attention`，PR 里必须说明你的改动与上游库的关系，以及是否需要同步回上游或已沟通 |
| **引用官方 roadmap，让 reviewer 知道你踩在主线上** | vLLM blog (Qwen3 NEXT 发布) | blog 明确写了 roadmap 中有 "Further kernel optimizations for GatedDeltaNet layers"。在 motivation 里引用这一条，reviewer 能快速确认这不是偏门贡献，属于团队已在规划的方向，有助于加快 review 优先级 |
| **Qwen3.5 和 Qwen3 NEXT 是两个分离的文件，但共享同一 kernel** | Issue #35924 | Issue #35924 清楚地展示了 `qwen3_5.py` 的 `Qwen3_5GatedDeltaNet` 和 `qwen3_next.py` 的 `Qwen3NextGatedDeltaNet` 各自独立，但底层都调同一个 `fused_recurrent_gated_delta_rule`。PR 里必须说清楚你改的是 kernel 层（两个模型都受益）还是某个 model 文件的调用层（只影响一个模型） |
| **GDN kernel 数值精度已知敏感，correctness 要格外严格** | vLLM blog (Qwen3 NEXT 发布) | Flash Linear Attention 团队专门参与过 numerics 改进，说明这个 kernel 在精度上有过坑。你的 correctness 部分 atol/rtol 数值要写出来，dtype 覆盖至少 bf16 + fp16 都要测，不能只测一种 dtype |
| **ROCm 路径用 Triton 跑 GDN，改了 triton config 需确认不破坏** | AMD 官方技术文章 | AMD 明确说 "The Gated Delta Networks in Qwen 3.5 are supported in vLLM via Triton-based kernels (`fused_recurrent_gated_delta_rule`). Since SGLang and vLLM support Triton on ROCm, these kernels work out-of-the-box on AMD GPU." 如果你改了 autotune config 或 kernel 逻辑，checklist 里要确认 ROCm 路径没被破坏；或在 PR 里明确说明本次优化仅针对 CUDA（给出理由，如"autotune configs are hardware-specific and ROCm falls back to safe defaults"） |
| **标题在 review 期间可以改，不用追求一次完美** | PR #14431 review history | PR #14431 的标题在 review 过程中由 maintainer 主导改了 3 次（从描述 kernel 名到描述优化目标到最终版本）。开 PR 时标题大方向对就行，内容和数据准确才是关键 |

---

## 五、参考 PR 链接及参考价值

| PR | 标题 | 重点参考内容 |
|----|------|------------|
| [#16173](https://github.com/vllm-project/vllm/pull/16173) | `[Kernel] support merge_attn_states CUDA kernel, 3x speedup` | **结构最完整的参考**：标题写 speedup；kernel 表格格式（tokens/heads/head_size/dtype/device/before/after/speedup，多 dtype 覆盖）；correctness 贴完整 pytest 输出含 `max abs diff`；被顺利 merge 的完整 description 结构 |
| [#14431](https://github.com/vllm-project/vllm/pull/14431) | `[Kernel] [V1] Further optimizations to ROCm (Triton) Backend to better handle GQA` | **TLDR 写法的最佳示范**（第一行加粗结论 + 数字）；`benchmark_serving.py` 完整输出格式（含所有指标字段）；review 过程中被要求补 accuracy test 的真实案例（教训来源） |
| [#14152](https://github.com/vllm-project/vllm/pull/14152) | `[Kernel] Improved performance for V1 Triton (ROCm) backend` | `lm_eval` accuracy 验证的写法；autotune 策略 tradeoff 的讨论方式 |
| [#22593](https://github.com/vllm-project/vllm/pull/22593) | `[Bugfix][Kernel] Support partial rotary embedding for MRoPE triton kernel` | 多前缀组合的标题写法；PR checklist 完整示例；reviewer 交互过程 |
| [#34110](https://github.com/vllm-project/vllm/pull/34110) | Qwen3.5 model family support (GDN 初始实现) | GDN 在 vLLM 里的原始实现上下文；理解 `qwen3_5.py` / `qwen3_next.py` 的 kernel 调用结构 |

---

## 六、给 Claude Code 的操作步骤

1. **读取改动文件**，确认：
   - kernel 文件路径（`vllm/attention/ops/` 下的具体文件名）
   - 改动涉及哪些参数（`BLOCK_SIZE_*`、`num_warps`、`num_stages`、autotune configs 等）
   - 是否修改了 `qwen3_5.py` / `qwen3_next.py` 的调用层（还是只改 kernel 文件）

2. **填充模板中所有 `[TODO: ...]`**，包括：
   - TLDR 里的加粗数字（从 benchmark 数据取最有代表性的值）
   - kernel 文件路径和行号
   - microbenchmark 表格（多行，覆盖不同 tokens / dtype 组合）
   - E2E benchmark 完整输出块（before & after 两块，保留所有字段）
   - correctness pytest 输出（含 max abs diff 数值，多 dtype）
   - test environment 版本信息（hardware / vLLM commit / Triton / PyTorch / CUDA）
   - upstream attribution 选择哪条说明

3. **PR 标题 speedup 数字的选取原则**：
   - 取常见生产 workload 下（中等 batch size、中等 seq len）的 speedup 数值
   - 不要取极端情况的峰值（会被 reviewer 质疑不代表实际场景）
   - 如果不同配置下收益差异大，可以写 "up to Xx speedup" 并在 description 里说明适用条件

4. **ROCm 一项的处理**：
   - 如果改动仅调整了 CUDA-specific autotune config（未改 kernel 逻辑）：写 "ROCm path unaffected; autotune configs are CUDA-specific and ROCm falls back to safe defaults"
   - 如果改了 kernel 逻辑：需要实际在 ROCm 环境验证，或在 PR 里明确说明范围限于 CUDA 并给出理由
```

