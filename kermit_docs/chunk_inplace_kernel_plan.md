# GDN Chunk-Mode In-Place Kernel Plan

## 0. Prerequisites — READ THIS FIRST

**Previous conversation context** ([full transcript](../../.claude/projects/-home-kermit-MyCode-vllm/c71c4f7d-1079-4f98-abf1-08d1c52f2eec.jsonl)):
- User wants in-place `ssm_state_indices` support for the GDN chunk attention path
- Two code paths exist: chunk (prefill, T>1) and recurrent (decode, T=1)
- Recurrent already supports `ssm_state_indices` + in-place state; chunk does NOT
- Chunk path does expensive gather (`ssm_state[indices].contiguous()`) and scatter (`ssm_state[indices] = ...`)
- Only 1 of 6 sub-kernels (`chunk_delta_h.py`) touches state — that's the only kernel to modify
- Rejected approaches: merging 6 sub-kernels, rewriting from scratch, modifying conv1d

**User's confirmed design decisions**:
- **Fusion scope**: Minimal — keep 6 sub-kernel architecture, only modify `chunk_delta_h.py`
- **Decode path**: Reuse existing `fused_recurrent_gated_delta_rule()` for T=1 sequences
- **Conv1d**: NOT included — caller handles it, same as today
- **API approach**: Modify EXISTING `chunk_gated_delta_rule()` — add `ssm_state_indices` parameter, no new function
- **H != HV for the MODEL**: Q/K heads (H_q) != value heads (HV). BUT in the chunk pipeline, `H` variable = `v.shape[-2]` = HV. So kernel's H == ssm_state's HV — mapping is 1:1.

---

## 1. Background & Motivation

### 1.1 Problem Statement

The chunk prefill path in GDN linear attention does unnecessary gather/scatter of state tensors.

**Current chunk path** (caller in `gdn_linear_attn.py` / `olmo_hybrid.py`):
```python
# Lines 949-971, gdn_linear_attn.py:
initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # ← GATHER
initial_state[~has_initial_state, ...] = 0                              # ← ZERO new seqs
o, last_recurrent_state = self.chunk_gated_delta_rule(
    q=query_non_spec, k=key_non_spec, v=value_non_spec,
    g=g_non_spec, beta=beta_non_spec,
    initial_state=initial_state,
    output_final_state=True,
    cu_seqlens=non_spec_query_start_loc,
    chunk_indices=attn_metadata.chunk_indices,
    chunk_offsets=attn_metadata.chunk_offsets,
    use_qk_l2norm_in_kernel=False,
)
ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)  # ← SCATTER
```

**Recurrent decode path** (already in-place, for reference):
```python
# Already uses ssm_state_indices, no gather/scatter:
core_attn_out_non_spec, _ = fused_sigmoid_gating_delta_rule_update(
    ...,
    initial_state=ssm_state,         # full cache, not gathered
    inplace_final_state=True,        # writes back to ssm_state directly
    ssm_state_indices=non_spec_state_indices_tensor,
)
```

The gather allocates `O(N * H * V * K)` temporary memory and copies from the pre-allocated cache. The scatter copies back. For large models (N=32, H=16, V=128, K=128) this is ~32 MB per layer wasted on redundant copies.

### 1.2 Goal

Modify `chunk_gated_delta_rule()` to accept `ssm_state_indices` so that:
1. State is read/written **in-place** through the pre-allocated `ssm_state` cache (no gather/scatter)
2. Callers pass `ssm_state` directly as `initial_state` + pass `ssm_state_indices`

### 1.3 Non-Goals

- NOT merging the 6 chunk sub-kernels
- NOT including causal_conv1d (caller keeps it)
- NOT rewriting the recurrent path (already supports ssm_state_indices)

---

## 2. Critical Technical Finding: H vs HV

**This is the most important section to understand before coding.**

### 2.1 The naming confusion

In the MODEL:
- Query heads (H_q): number of query attention heads
- Key heads (Hg): number of key/gate heads (GQA reduces these)
- Value heads (HV): number of value heads
- `ssm_state` shape: `[max_blocks, HV, V, K]`

In the CHUNK PIPELINE, `recompute_w_u_fwd` (wy_fast.py line 113) redefines H:
```python
H = v.shape[-2]  # This is HV, NOT H_q!
```

So the chunk pipeline's `H` variable = model's HV = value head count. All subsequent kernels use this H.

### 2.2 What this means for the kernel

`chunk_delta_h.py` kernel:
- Grid: `(triton.cdiv(V, BV), N * H)` where H = HV = value head count
- `i_nh = i_n * H + i_h` — iterates over value heads
- `initial_state` shape: `[N, H, V, K]` where H = HV
- `ssm_state` shape: `[max_blocks, H, V, K]` — SAME H!
- **Mapping is 1:1** — no head folding needed!

### 2.3 The in-place pointer formula (VERIFIED)

```
Given:
  ssm_state shape: [max_blocks, H, V, K]
  stride_init_state_token = ssm_state.stride(0) = H * V * K
  i_nh = i_n * H + i_h  (flat head index)

Current (non-CB) — line 105 of chunk_delta_h.py:
  h0 = initial_state.data_ptr() + i_nh * V * K
      = initial_state.data_ptr() + (i_n * H + i_h) * V * K
      = initial_state[i_n, i_h, :, :]  ✓
  (initial_state has shape [N, H, V, K])

For CB (in-place):
  initial_state IS ssm_state (shape [max_blocks, H, V, K])
  We need: ssm_state[state_idx, i_h, :, :]
         = ssm_state.data_ptr() + state_idx * H * V * K + i_h * V * K
  
  After line 105, h0 = ssm_state.data_ptr() + i_n * H * V * K + i_h * V * K
  Target:            ssm_state.data_ptr() + state_idx * H * V * K + i_h * V * K
  Delta = (state_idx - i_n) * H * V * K = (state_idx - i_n) * stride_init_state_token

FINAL FORMULA:  h0 += (state_idx - i_n) * stride_init_state_token  ✓
```

This works because the `i_n * H * V * K` term from the original offset cancels with `-i_n * stride_init_state_token`, leaving only `state_idx * H * V * K + i_h * V * K` — exactly `ssm_state[state_idx, i_h, :, :]`.

Same formula applies symmetrically to `ht` (final state storing).

---

## 3. Architecture Overview

```
Modified chunk_gated_delta_rule(q, k, v, g, beta, ..., ssm_state_indices=None)
│
│  (ssm_state_indices is None → existing behavior, backward compatible)
│  (ssm_state_indices is set → in-place path)
│
├─ chunk_local_cumsum     — UNCHANGED
├─ chunk_scaled_dot_kkt   — UNCHANGED
├─ solve_tril             — UNCHANGED
├─ recompute_w_u_fwd      — UNCHANGED
├─ chunk_delta_h_fwd_h    — ★ MODIFIED: ssm_state_indices for in-place state R/W
└─ chunk_fwd_o            — UNCHANGED
```

**Data flow change**:
```
BEFORE:
  ssm_state [max_blocks, HV, V, K]
    → gather via indices → initial_state [N, H, V, K] (contiguous copy)
    → chunk pipeline reads/writes initial_state/final_state
    → scatter via indices → ssm_state (copy back)

AFTER:
  ssm_state [max_blocks, HV, V, K]
    → chunk pipeline reads/writes ssm_state directly via indices
    → zero-copy, in-place
```

---

## 4. Detailed Implementation Steps

### Step 4.1: Modify `chunk_delta_h.py` kernel

**File**: `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`

#### 4.1.1 Add heuristic and kernel parameters

Add to the `@triton.heuristics` decorator (line 22-31):
```python
"IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
```

Add to kernel function signature (after `chunk_offsets` parameter):
```python
ssm_state_indices,            # [N] tensor, maps seq n → cache block ID
stride_init_state_token: tl.constexpr,   # ssm_state.stride(0) = H * V * K
stride_final_state_token: tl.constexpr,  # same as init (in-place)
stride_indices_seq: tl.constexpr,        # ssm_state_indices.stride(0)
IS_CONTINUOUS_BATCHING: tl.constexpr,
```

Add `IS_CONTINUOUS_BATCHING` to the autotune key:
```python
key=["H", "K", "V", "BT", "IS_CONTINUOUS_BATCHING"],
```

#### 4.1.2 Modify initial state loading (current lines 104-127)

**Current code:**
```python
if USE_INITIAL_STATE:
    h0 = h0 + i_nh * V * K              # line 105
# ... then block_ptr loads from h0 (lines 110-127), e.g.:
p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
# ... K > 64, 128, 192 blocks same pattern
```

**New code:**
```python
if USE_INITIAL_STATE:
    h0 = h0 + i_nh * V * K              # line 105 — KEEP UNCHANGED
    if IS_CONTINUOUS_BATCHING:
        state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq
        ).to(tl.int64)
        if state_idx > 0:
            # Apply verified formula from Section 2.3
            h0 = h0 + (state_idx - i_n) * stride_init_state_token
        else:
            # NULL_BLOCK_ID — skip loading, state stays zero
            pass  # falls through to guarded loads below

# Load initial state (same block_ptr code, now works for both CB and non-CB)
if USE_INITIAL_STATE:
    if IS_CONTINUOUS_BATCHING:
        # Guard: only load if state_idx was valid
        state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq
        ).to(tl.int64)
        should_load = state_idx > 0
    else:
        should_load = True

    if should_load:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        # ... K > 64, 128, 192 blocks — same pattern
```

**IMPORTANT**: The `state_idx` load happens twice (once for pointer adjustment, once for guard). The implementing agent may optimize this: load once into a variable and reuse. Triton should handle this fine with CSE, but being explicit is safer:
```python
if USE_INITIAL_STATE:
    h0 = h0 + i_nh * V * K
    should_load = True
    if IS_CONTINUOUS_BATCHING:
        state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq
        ).to(tl.int64)
        if state_idx > 0:
            h0 = h0 + (state_idx - i_n) * stride_init_state_token
        else:
            should_load = False
    if should_load:
        # ... block_ptr loads unchanged ...
```

#### 4.1.3 Modify final state storing (lines 281-298)

The `ht` pointer is offset identically to `h0` (line 107 in the current code does `ht = ht + i_nh * V * K` for the contiguous case).

**New code — apply the SAME verified formula:**
```python
if STORE_FINAL_STATE:
    if IS_CONTINUOUS_BATCHING:
        state_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq
        ).to(tl.int64)
        if state_idx > 0:
            ht = ht + (state_idx - i_n) * stride_final_state_token + i_nh * V * K
        else:
            # NULL_BLOCK_ID — skip storing
            pass
    else:
        ht = ht + i_nh * V * K  # existing code

    # Block ptr stores — identical for both CB and non-CB after pointer adjustment
    p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
    tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
    # ... K > 64, 128, 192 blocks — same pattern
```

#### 4.1.4 NULL_BLOCK_ID guard

For both initial state load and final state store, when `state_idx <= 0`:
- Initial state: skip loading, b_h register stays zero (correct for new/empty sequences)
- Final state: skip storing, don't write to invalid block

The cleanest way: set a flag and guard the loads/stores:

```python
if IS_CONTINUOUS_BATCHING:
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    valid_state = state_idx > 0
else:
    valid_state = True

if USE_INITIAL_STATE and valid_state:
    # ... load initial state
```

### Step 4.2: Modify `chunk_gated_delta_rule_fwd_h` wrapper

**File**: `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` (same file, function at lines 301-361)

#### 4.2.1 Add parameters
```python
def chunk_gated_delta_rule_fwd_h(
    k, w, u,
    g=None, gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=FLA_CHUNK_SIZE,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_offsets=None,
    ssm_state_indices=None,       # ★ NEW
):
```

#### 4.2.2 Compute strides
```python
if ssm_state_indices is not None:
    stride_indices_seq = ssm_state_indices.stride(0)
    stride_init_state_token = initial_state.stride(0)   # H * V * K
    stride_final_state_token = initial_state.stride(0)   # same (in-place)
else:
    stride_indices_seq = 1
    stride_init_state_token = 1
    stride_final_state_token = 1
```

#### 4.2.3 Modify final_state allocation (lines 333-335)
```python
if ssm_state_indices is not None:
    # In-place: final_state IS initial_state (the full ssm_state cache)
    # The kernel writes back via indices to the correct blocks
    final_state = initial_state if output_final_state else None
else:
    # Existing behavior: allocate new contiguous tensor
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
```

#### 4.2.4 Pass new args to kernel
```python
chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
    k=k, v=u, w=w, v_new=v_new,
    g=g, gk=gk,
    h=h, h0=initial_state, ht=final_state,
    cu_seqlens=cu_seqlens,
    chunk_offsets=chunk_offsets,
    T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
    ssm_state_indices=ssm_state_indices,                     # ★ NEW
    stride_indices_seq=stride_indices_seq,                   # ★ NEW
    stride_init_state_token=stride_init_state_token,         # ★ NEW
    stride_final_state_token=stride_final_state_token,       # ★ NEW
)
```

### Step 4.3: Modify `chunk.py` — add ssm_state_indices pass-through

**File**: `vllm/model_executor/layers/fla/ops/chunk.py`

#### 4.3.1 Modify `chunk_gated_delta_rule_fwd` (the internal pipeline function, lines 25-58)

Add `ssm_state_indices` parameter and pass through to `chunk_gated_delta_rule_fwd_h`:
```python
def chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale,
    initial_state, output_final_state,
    cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
    ssm_state_indices=None,  # ★ NEW
):
    # ... existing pipeline code unchanged except:
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        ssm_state_indices=ssm_state_indices,  # ★ NEW
    )
    # ... rest unchanged
```

#### 4.3.2 Modify `ChunkGatedDeltaRuleFunction.forward` (lines 72-101)

Add `ssm_state_indices` to the autograd function:
```python
@staticmethod
def forward(
    ctx, q, k, v, g, beta, scale,
    initial_state, output_final_state,
    cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
    use_qk_l2norm_in_kernel=False,
    ssm_state_indices=None,  # ★ NEW
):
    # ... existing code unchanged except pass through
    g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        ssm_state_indices=ssm_state_indices,  # ★ NEW
    )
    return o.to(q.dtype), final_state
```

#### 4.3.3 Modify `chunk_gated_delta_rule` (public API, lines 120-210)

Add `ssm_state_indices` parameter:
```python
@torch.compiler.disable
def chunk_gated_delta_rule(
    q, k, v, g, beta,
    scale=None, initial_state=None,
    output_final_state=False,
    cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
    use_qk_l2norm_in_kernel=False,
    ssm_state_indices=None,  # ★ NEW
):
    # ... validation unchanged ...
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale,
        initial_state, output_final_state,
        cu_seqlens, chunk_indices, chunk_offsets,
        use_qk_l2norm_in_kernel,
        ssm_state_indices,  # ★ NEW
    )
    return o, final_state
```

### Step 4.4: Update callers

**File**: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
**File**: `vllm/model_executor/models/olmo_hybrid.py`

#### 4.4.1 gdn_linear_attn.py (lines ~947-971)

**BEFORE:**
```python
if attn_metadata.num_prefills > 0:
    assert non_spec_state_indices_tensor is not None
    initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
    assert has_initial_state is not None
    initial_state[~has_initial_state, ...] = 0
    (
        core_attn_out_non_spec,
        last_recurrent_state,
    ) = self.chunk_gated_delta_rule(
        q=query_non_spec, k=key_non_spec, v=value_non_spec,
        g=g_non_spec, beta=beta_non_spec,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=non_spec_query_start_loc,
        chunk_indices=attn_metadata.chunk_indices,
        chunk_offsets=attn_metadata.chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
```

**AFTER:**
```python
if attn_metadata.num_prefills > 0:
    assert non_spec_state_indices_tensor is not None
    # Zero out state for NEW sequences (no prior KV state)
    if has_initial_state is not None:
        zero_mask = ~has_initial_state
        if zero_mask.any():
            zero_indices = non_spec_state_indices_tensor[zero_mask]
            ssm_state[zero_indices] = 0
    
    # In-place chunk: passes ssm_state directly; no gather/scatter
    core_attn_out_non_spec, _ = self.chunk_gated_delta_rule(
        q=query_non_spec, k=key_non_spec, v=value_non_spec,
        g=g_non_spec, beta=beta_non_spec,
        initial_state=ssm_state,           # ← full cache, NOT gathered
        output_final_state=True,
        cu_seqlens=non_spec_query_start_loc,
        chunk_indices=attn_metadata.chunk_indices,
        chunk_offsets=attn_metadata.chunk_offsets,
        use_qk_l2norm_in_kernel=False,
        ssm_state_indices=non_spec_state_indices_tensor,  # ★ NEW
    )
    # No scatter — state already updated in-place
```

#### 4.4.2 olmo_hybrid.py (lines ~551-568)

Same pattern as gdn_linear_attn.py. Replace the gather→chunk→scatter with direct in-place call. The exact line numbers should be verified by reading the file.

### Step 4.5: Handle the `ChunkGatedDeltaRuleFunction` autograd backward

The `ChunkGatedDeltaRuleFunction` is a `torch.autograd.Function` used only for forward pass (inference). In `chunk.py`:
```python
class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, ...):
        ...
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state
```

There is NO `backward` method — this is inference-only. So adding `ssm_state_indices` to the forward signature is safe.

---

## 5. Key Reference Files

All paths relative to repo root:

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` | 1-362 | ★ ONLY KERNEL TO MODIFY |
| `vllm/model_executor/layers/fla/ops/chunk.py` | 1-235 | Parameter pass-through |
| `vllm/model_executor/layers/fla/ops/fused_recurrent.py` | 1-619 | Reference: IS_CONTINUOUS_BATCHING pattern |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 920-1010 | Caller 1 — replace gather/scatter |
| `vllm/model_executor/models/olmo_hybrid.py` | 540-570 | Caller 2 — replace gather/scatter |
| `vllm/v1/attention/backends/gdn_attn.py` | 1-476 | Metadata builder (NO changes needed) |
| `vllm/model_executor/layers/fla/ops/utils.py` | — | FLA_CHUNK_SIZE=64 |
| `vllm/v1/attention/backends/utils.py` | — | NULL_BLOCK_ID |
| `vllm/model_executor/layers/fla/ops/wy_fast.py` | 113 | H = v.shape[-2] — confirms H==HV in pipeline |

---

## 6. Step 0 — Verify Pointer Arithmetic Formula

**MANDATORY**: Before modifying any production code, write a minimal Triton kernel to verify the pointer arithmetic formula.

### 6.0.1 Verification kernel

Create `/tmp/verify_ptr_arith.py`:
```python
import torch
import triton
import triton.language as tl

@triton.jit
def verify_ptr_kernel(
    h0,                          # ssm_state [max_blocks, H, V, K]
    ssm_state_indices,           # [N]
    out_ptr,                     # [N, H, V, K] — debug output
    N, H, V, K,
    stride_init_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    BV = 8  # small for debugging

    # === SIMULATES LINE 105 ===
    h0 = h0 + i_nh * V * K  # same as current code

    if IS_CONTINUOUS_BATCHING:
        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
        # === APPLY FORMULA ===
        h0 = h0 + (state_idx - i_n) * stride_init_state_token

    # Load test value: ssm_state[..., 0, 0]
    test_val = tl.load(h0 + i_v * BV * K)  # first K element at (i_v*BV, 0)
    
    # Store to output for verification
    out_off = i_nh * V * K + i_v * BV * K
    tl.store(out_ptr + out_off, test_val)

def test_formula():
    max_blocks, N, H, V, K = 16, 4, 8, 128, 128
    device = 'cuda'
    
    # Setup: put known values into ssm_state
    ssm_state = torch.zeros(max_blocks, H, V, K, device=device)
    indices = torch.tensor([3, 7, 1, 5], device=device, dtype=torch.int32)
    
    # Write known pattern: block b, head h, row v, col k → encode in value
    for b in range(max_blocks):
        for h in range(H):
            ssm_state[b, h, :, :] = float(b * 1000 + h)  # unique per (block, head)

    # Reference: what should be loaded
    expected = torch.zeros(N, H, V, K, device=device)
    for n in range(N):
        state_idx = indices[n].item()
        for h in range(H):
            expected[n, h, :, :] = float(state_idx * 1000 + h)

    # Run kernel
    BV = 8
    grid = (triton.cdiv(V, BV), N * H)
    actual = torch.zeros(N, H, V, K, device=device)
    
    verify_ptr_kernel[grid](
        h0=ssm_state,
        ssm_state_indices=indices,
        out_ptr=actual,
        N=N, H=H, V=V, K=K,
        stride_init_state_token=ssm_state.stride(0),
        stride_indices_seq=indices.stride(0),
        IS_CONTINUOUS_BATCHING=True,
    )
    
    # Compare first element of each (n,h) pair
    for n in range(N):
        for h in range(H):
            exp_val = expected[n, h, 0, 0].item()
            act_val = actual[n, h, 0, 0].item()
            assert abs(exp_val - act_val) < 0.01, \
                f"Mismatch at n={n}, h={h}: expected={exp_val}, got={act_val}"
    
    print("✓ Pointer arithmetic formula VERIFIED")

if __name__ == "__main__":
    test_formula()
```

Run: `python /tmp/verify_ptr_arith.py`

Expected output: `✓ Pointer arithmetic formula VERIFIED`

### 6.0.2 What this validates

1. The formula `h0 += (state_idx - i_n) * stride_init_state_token` correctly translates from contiguous `[N, H, V, K]` indexing to sparse `[max_blocks, H, V, K]` indexing
2. The `i_n` term cancels correctly with the existing `i_nh = i_n * H + i_h` offset
3. Different state_idx values for different sequences don't interfere
4. The block_ptr loads after the adjustment hit the right memory locations

**If this test FAILS**: DO NOT proceed to kernel modification. Debug the formula first. Check that:
- `stride_init_state_token == H * V * K` (ssm_state is contiguous)
- `ssm_state_indices` has the correct dtype (int32)
- The arithmetic is integer-exact in Triton

---

## 7. Implementation Order

1. **Read ALL reference files** listed in Section 5. Understand every line of chunk_delta_h.py.
2. **Trace the H == HV proof**: In wy_fast.py line 113, H = v.shape[-2]. In chunk.py line 315 (chunk_delta_h wrapper), H = u.shape[-2] where u comes from recompute_w_u_fwd. Verify this equals HV.
3. **Modify chunk_delta_h.py kernel**: Add IS_CONTINUOUS_BATCHING, ssm_state_indices, and in-place state load/store using the verified formula `h0 += (state_idx - i_n) * stride_init_state_token`.
4. **Modify chunk_delta_h.py wrapper**: Add ssm_state_indices parameter, compute strides, pass to kernel, handle final_state=initial_state for in-place mode.
5. **Modify chunk.py**: Thread ssm_state_indices through all 3 layers (chunk_gated_delta_rule_fwd, ChunkGatedDeltaRuleFunction.forward, chunk_gated_delta_rule).
6. **Update gdn_linear_attn.py**: Replace gather/scatter with direct in-place call. Handle initial state zeroing for new sequences.
7. **Update olmo_hybrid.py**: Same change.
8. **Write tests** (see Section 7).
9. **Benchmark** (see Section 8).

---

## 7. Testing Plan

### 7.1 Unit test — correctness

```python
# File: tests/kernels/test_chunk_inplace.py
import torch
import pytest
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

def test_chunk_ssm_state_indices_correctness():
    """Verify in-place (ssm_state_indices) produces identical results to gather/scatter."""
    N, T_per_seq, H, K, V = 4, 256, 8, 128, 128
    max_blocks = 16
    total_T = N * T_per_seq
    device = 'cuda'
    dtype = torch.bfloat16

    # Create inputs in varlen format [1, total_T, ...]
    q = torch.randn(1, total_T, H, K, device=device, dtype=dtype)
    k = torch.randn(1, total_T, H, K, device=device, dtype=dtype)
    v = torch.randn(1, total_T, H, V, device=device, dtype=dtype)
    g = torch.randn(1, total_T, H, device=device, dtype=torch.float32)
    beta = torch.randn(1, total_T, H, device=device, dtype=torch.float32)
    cu_seqlens = torch.arange(0, total_T + 1, T_per_seq, device=device, dtype=torch.int32)

    # Pre-allocated ssm_state cache
    ssm_state = torch.randn(max_blocks, H, V, K, device=device, dtype=torch.float32)
    indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

    # --- Reference: gather/scatter path ---
    ssm_ref = ssm_state.clone()
    initial_state_ref = ssm_ref[indices].contiguous().clone()
    o_ref, final_ref = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=initial_state_ref,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ssm_ref[indices] = final_ref.to(ssm_ref.dtype)

    # --- Test: in-place path ---
    ssm_ip = ssm_state.clone()
    o_ip, final_ip = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=ssm_ip,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )

    # final_ip should be the same object as ssm_ip (in-place)
    assert final_ip is ssm_ip, "final_state should be the same tensor as ssm_state (in-place)"

    # Numeric verification
    torch.testing.assert_close(o_ref, o_ip, atol=1e-2, rtol=1e-2,
        msg="Output mismatch between gather/scatter and in-place")
    torch.testing.assert_close(ssm_ref, ssm_ip, atol=1e-2, rtol=1e-2,
        msg="ssm_state mismatch between gather/scatter and in-place")
```

### 7.2 Unit test — NULL_BLOCK_ID handling

```python
def test_chunk_null_block_id():
    """Verify NULL_BLOCK_ID entries are no-ops (no crash, correct output)."""
    # Same setup but with some indices = NULL_BLOCK_ID (0 or -1)
    # Verify those sequences produce correct output (state treated as zero)
```

### 7.3 Unit test — backward compatibility

```python
def test_chunk_backward_compat():
    """Without ssm_state_indices, behavior must be identical to before."""
    # Old API (no ssm_state_indices) must work unchanged
```

### 7.4 Integration test

```python
# File: tests/models/test_gdn_inplace_chunk.py
def test_gdn_model_inplace_chunk():
    """End-to-end: load GDN model, run prefill with in-place chunk, compare to reference."""
    # 1. Load a small model (Qwen3-Next or similar with GDN layers)
    # 2. Create a test batch with prefill sequences
    # 3. Run with OLD gather/scatter → save outputs
    # 4. Run with NEW in-place → compare outputs
    # 5. Verify ssm_state cache contents match
```

---

## 8. Performance Benchmarking

### 8.1 Memory

Run with `torch.cuda.memory_stats()` before and after the chunk call:

| Metric | Before | After |
|--------|--------|-------|
| Peak allocated | `O(N * H * V * K)` temp tensor | 0 extra |
| Example (N=8,H=16,V=128,K=128) | ~8 MB/layer | 0 |

### 8.2 Latency

Use `torch.cuda.Event` to measure the prefill path:
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... chunk call ...
end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)
```

Expected: 5-15% reduction in prefill path latency from eliminating memory copies.

---

## 9. Gotchas & Pitfalls

1. **Pointer arithmetic is tricky**: The formula `h0 += (state_idx - i_n) * stride_init_state_token` MUST be verified against actual Triton behavior. Write a tiny test kernel first that just loads/stores state.

2. **`initial_state` dtype**: ssm_state is `float32`, but model activations are `bfloat16`. The kernel already casts to float32 when loading state. No change needed.

3. **Initial state zeroing**: New sequences must have their cache entries zeroed BEFORE the kernel runs. The caller (gdn_linear_attn.py) must do `ssm_state[new_seq_indices] = 0`. This replaces the old `initial_state[~has_initial_state, ...] = 0`.

4. **CUDAGraph compatibility**: The kernel must produce identical output tensors regardless of `IS_CONTINUOUS_BATCHING` value — CUDA graphs capture based on static shapes. The `final_state` tensor shape differs:
   - Old: `[N, H, V, K]` (new allocation)
   - New: `[max_blocks, H, V, K]` (ssm_state itself)
   - Callers that use `last_recurrent_state` must handle this difference. Check `build_for_cudagraph_capture` in gdn_attn.py.

5. **Autotune configs**: Adding `IS_CONTINUOUS_BATCHING` to the key doubles compiled configs → longer JIT compile time. Consider whether it actually affects kernel performance. If CB branch adds minimal overhead, keep it out of the key.

6. **`is_signed` for indices**: `ssm_state_indices` uses `int32`. Make sure `tl.load` with conversion to `int64` handles negative values correctly. NULL_BLOCK_ID is typically 0 and some padding entries, so check `<= 0`.

7. **Variable-length + CB interaction**: The kernel already handles both `IS_VARLEN` (cu_seqlens) and needs to also handle `IS_CONTINUOUS_BATCHING` (ssm_state_indices). These are orthogonal — both can be True simultaneously.

---

## 10. Code Locations Summary

```
Files to MODIFY:
  vllm/model_executor/layers/fla/ops/chunk_delta_h.py
    - Line 22-31:   Add IS_CONTINUOUS_BATCHING heuristic
    - Line 33-41:   Add to autotune key
    - Line 43-68:   Add kernel parameters (ssm_state_indices, strides, IS_CONTINUOUS_BATCHING)
    - Line 104-127: Modify initial state loading (CB branch)
    - Line 281-298: Modify final state storing (CB branch)
    - Line 301-361: Modify wrapper function (params, strides, final_state alloc)
  vllm/model_executor/layers/fla/ops/chunk.py
    - Line 25-58:   Add ssm_state_indices to chunk_gated_delta_rule_fwd
    - Line 72-101:  Add ssm_state_indices to ChunkGatedDeltaRuleFunction.forward
    - Line 120-210: Add ssm_state_indices to chunk_gated_delta_rule public API
  vllm/model_executor/layers/mamba/gdn_linear_attn.py
    - Line 948-971: Replace gather/scatter with in-place call
  vllm/model_executor/models/olmo_hybrid.py
    - Line 540-568: Replace gather/scatter with in-place call

Files to READ (for understanding, NO changes):
  vllm/model_executor/layers/fla/ops/fused_recurrent.py  (CB reference pattern)
  vllm/model_executor/layers/fla/ops/wy_fast.py           (H=HV proof)
  vllm/model_executor/layers/fla/ops/utils.py             (constants)
  vllm/v1/attention/backends/gdn_attn.py                  (metadata)
  vllm/v1/attention/backends/utils.py                     (NULL_BLOCK_ID)

New files to CREATE:
  tests/kernels/test_chunk_inplace.py                      (unit tests)
  tests/models/test_gdn_inplace_chunk.py                   (integration tests)
```
