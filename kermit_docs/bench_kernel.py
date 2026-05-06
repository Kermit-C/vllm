#!/usr/bin/env python3
"""
Kernel-level microbenchmark for chunk_gated_delta_rule.

Measures pure kernel execution time using synthetic data matching target model
dimensions. No model loading required. Uses CUDA event timing.

Usage:
    conda activate vllm-20
    cd /home/kermit/MyCode/vllm

    # On main branch (baseline):
    git checkout main
    PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen --output /tmp/kernel_main.json

    # On feature branch (optimized):
    git checkout feature/gdn-prefill-kernal-opt
    PYTHONPATH=. python kermit_docs/bench_kernel.py --dims qwen --output /tmp/kernel_feat.json

    # Compare:
    python -c "
    import json
    m=json.load(open('/tmp/kernel_main.json'))
    f=json.load(open('/tmp/kernel_feat.json'))
    for k in m["scenarios"]:
        dm = m["scenarios"][k]['avg_us'] - f["scenarios"][k]['avg_us']
        pct = dm / m["scenarios"][k]['avg_us'] * 100
        print(f'{k:<24s}  main={m["scenarios"][k]["avg_us"]:8.1f}us  feat={f["scenarios"][k]["avg_us"]:8.1f}us  Δ={dm:+7.1f}us  ({pct:+.1f}%)')
    "

    # OLMo model dims:
    PYTHONPATH=. python kermit_docs/bench_kernel.py --dims olmo --output /tmp/kernel_olmo.json

Environment:
    GPU: any CUDA GPU
    Conda: vllm-20
"""

import argparse
import inspect
import json
import os
import sys

import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

_USE_SSM_STATE_INDICES = 'ssm_state_indices' in inspect.signature(chunk_gated_delta_rule).parameters

# ── Model dimension presets ──────────────────────────────────────────────


def get_dims(dims_name: str) -> dict:
    """Return (H_k, HV, K, V) for the target model's GDN linear attention layers."""
    presets = {
        "qwen": {
            "num_k_heads": 16,
            "num_v_heads": 32,
            "key_head_dim": 128,
            "value_head_dim": 128,
        },
        "olmo": {
            "num_k_heads": 30,
            "num_v_heads": 30,
            "key_head_dim": 96,
            "value_head_dim": 192,
        },
        "qwen0.8b": {
            "num_k_heads": 16,
            "num_v_heads": 16,
            "key_head_dim": 128,
            "value_head_dim": 128,
        },
    }
    if dims_name not in presets:
        raise ValueError(f"Unknown dims '{dims_name}'. Choose: {list(presets.keys())}")
    return presets[dims_name]


# ── Benchmark scenarios ──────────────────────────────────────────────────

SCENARIOS = [
    # (name, N, T_per_seq) — N=seqs, T=tokens per seq
    # Prefill
    ("prefill_T64", 1, 64),
    ("prefill_T128", 1, 128),
    ("prefill_T256", 1, 256),
    ("prefill_T512", 1, 512),
    ("prefill_T1024", 1, 1024),
    # Decode (T=1, varying batch)
    ("decode_N1", 1, 1),
    ("decode_N16", 16, 1),
    ("decode_N64", 64, 1),
    ("decode_N128", 128, 1),
    # Mixed
    ("mixed_2pf_14d", 16, 16),  # 2 prefill tokens + 14 decode → avg T=2
    ("mixed_4pf_28d", 32, 16),  # 4 seqs × T=16 → some long, some short
    ("mixed_8pf_56d", 64, 16),
]

WARMUP = 30
REPEAT = 100


def make_inputs(N, T_per_seq, dims, device="cuda", dtype=torch.bfloat16):
    """Create synthetic inputs for chunk_gated_delta_rule matching target model dims."""
    H_k = dims["num_k_heads"]
    HV = dims["num_v_heads"]
    K = dims["key_head_dim"]
    V = dims["value_head_dim"]

    total_T = N * T_per_seq
    max_blocks = max(256, N + 16)
    scale = K**-0.5

    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    q = torch.randn(1, total_T, H_k, K, device=device, dtype=dtype, generator=gen)
    k = torch.nn.functional.normalize(
        torch.randn(1, total_T, H_k, K, device=device, dtype=dtype, generator=gen),
        p=2,
        dim=-1,
    )
    v = 0.1 * torch.randn(1, total_T, HV, V, device=device, dtype=dtype, generator=gen)
    g = torch.nn.functional.logsigmoid(
        torch.rand(1, total_T, H_k, device=device, dtype=torch.float32, generator=gen)
    )
    beta = torch.rand(1, total_T, H_k, device=device, dtype=torch.float32, generator=gen).sigmoid()
    cu_seqlens = torch.arange(0, total_T + 1, T_per_seq, device=device, dtype=torch.int32)

    # State pool (simulates real serving's shared state pool)
    state_pool = torch.randn(max_blocks, HV, V, K, device=device, dtype=torch.float32, generator=gen)
    # Random indices: which pool slots each sequence maps to
    state_indices = torch.randperm(max_blocks, device=device, generator=gen)[:N].to(torch.int32)
    # ~50% of sequences have prior state (simulates has_initial_state)
    has_initial_state = torch.randint(0, 2, (N,), device=device, dtype=torch.bool, generator=gen)

    kernel_indices = torch.arange(1, N + 1, device=device, dtype=torch.int32) if _USE_SSM_STATE_INDICES else None

    return q, k, v, g, beta, cu_seqlens, state_pool, state_indices, has_initial_state, kernel_indices, scale, max_blocks


def bench_scenario(name, N, T, dims):
    """Run one benchmark scenario, return (avg_us, min_us, median_us).

    Matches real serving paths from gdn_linear_attn.py:
    - Main branch (no ssm_state_indices): gather + zero + kernel + scatter
    - Feature branch (with ssm_state_indices): kernel directly on pool
    """
    q, k, v, g, beta, cu_seqlens, state_pool, state_indices, has_initial_state, kernel_indices, scale, max_blocks = make_inputs(N, T, dims)
    is_prefill = T > 1

    def run_once(pool):
        if is_prefill:
            if kernel_indices is not None:
                # Feature branch: kernel handles indexing + zeroing in-place
                chunk_gated_delta_rule(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale, initial_state=pool,
                    output_final_state=True, cu_seqlens=cu_seqlens,
                    ssm_state_indices=state_indices,
                    has_initial_state=has_initial_state,
                )
            else:
                # Main branch: manual gather + zero + kernel + scatter
                initial_state = pool[state_indices].contiguous()
                initial_state[~has_initial_state, ...] = 0
                _, last_state = chunk_gated_delta_rule(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale, initial_state=initial_state,
                    output_final_state=True, cu_seqlens=cu_seqlens,
                )
                pool[state_indices] = last_state.to(pool.dtype)
        else:
            ssm = pool[:N].clone() if kernel_indices is None else pool
            kwargs = dict(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale, initial_state=ssm, output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
            if kernel_indices is not None:
                kwargs['ssm_state_indices'] = kernel_indices
            chunk_gated_delta_rule(**kwargs)

    # Warmup (compile Triton kernels)
    for _ in range(WARMUP):
        run_once(state_pool.clone())
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    measurements = []
    for _ in range(REPEAT):
        pool = state_pool.clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once(pool)
        end.record()
        torch.cuda.synchronize()
        measurements.append(start.elapsed_time(end) * 1000)  # ms → μs

    measurements.sort()
    avg = sum(measurements) / len(measurements)
    median = measurements[len(measurements) // 2]
    return avg, min(measurements), median


def main():
    parser = argparse.ArgumentParser(description="Kernel-level microbenchmark for chunk_gated_delta_rule")
    parser.add_argument("--dims", default="qwen", choices=["qwen", "qwen0.8b", "olmo"],
                        help="Model dimensions preset (default: qwen)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--warmup", type=int, default=WARMUP, help=f"Warmup iterations (default: {WARMUP})")
    parser.add_argument("--repeat", type=int, default=REPEAT, help=f"Benchmark iterations (default: {REPEAT})")
    args = parser.parse_args()

    dims = get_dims(args.dims)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name} ({props.total_memory // 1024**3}GB)")
    print(f"Dims: H_k={dims['num_k_heads']} HV={dims['num_v_heads']} "
          f"K={dims['key_head_dim']} V={dims['value_head_dim']}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print(f"{'='*80}")

    results = {
        "meta": {
            "dims": args.dims,
            "num_k_heads": dims["num_k_heads"],
            "num_v_heads": dims["num_v_heads"],
            "key_head_dim": dims["key_head_dim"],
            "value_head_dim": dims["value_head_dim"],
            "warmup": args.warmup,
            "repeat": args.repeat,
            "gpu": props.name,
        },
        "scenarios": {},
    }

    for name, N, T in SCENARIOS:
        avg, min_val, median = bench_scenario(name, N, T, dims)
        results["scenarios"][name] = {
            "N": N,
            "T": T,
            "avg_us": round(avg, 1),
            "min_us": round(min_val, 1),
            "median_us": round(median, 1),
        }
        print(f"  {name:<24s}  N={N:3d}  T={T:4d}  "
              f"avg={avg:8.1f}μs  min={min_val:8.1f}μs  median={median:8.1f}μs")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
