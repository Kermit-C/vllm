#!/usr/bin/env python3
"""Test 1: Precision comparison — OLD gather/scatter vs NEW in-place kernel.

Compares ssm_state (strict) and output (tolerance) across configs matching
Qwen3.5-9B model dimensions.
"""
import os, sys

_conda_bin = "/home/kermit/.conda/envs/vllm-20/bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch


def setup_inputs(N, T_per_seq, H, V, K, max_blocks, device, dtype, seed):
    total_T = N * T_per_seq
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    q = 0.1 * torch.randn(1, total_T, H, K, device=device, dtype=dtype, generator=gen)
    k = torch.nn.functional.normalize(
        torch.randn(1, total_T, H, K, device=device, dtype=dtype, generator=gen),
        p=2, dim=-1,
    )
    v = 0.1 * torch.randn(1, total_T, H, V, device=device, dtype=dtype, generator=gen)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total_T, H, device=device, dtype=torch.float32, generator=gen)
    )
    beta = torch.rand(1, total_T, H, device=device, dtype=torch.float32, generator=gen).sigmoid()
    cu_seqlens = torch.arange(0, total_T + 1, T_per_seq, device=device, dtype=torch.int32)
    scale = K ** -0.5

    ssm_state = torch.randn(max_blocks, H, V, K, device=device, dtype=torch.float32, generator=gen)
    indices = torch.arange(1, N + 1, device=device, dtype=torch.int32)
    has_state = torch.ones(N, device=device, dtype=torch.bool)

    return q, k, v, g, beta, scale, cu_seqlens, ssm_state, indices, has_state


def run_old_path(q, k, v, g, beta, scale, cu_seqlens, ssm_state, indices, has_init):
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    ssm = ssm_state.clone()
    init = ssm[indices].contiguous()
    init[~has_init] = 0
    o, final_state = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale,
        initial_state=init, output_final_state=True, cu_seqlens=cu_seqlens,
    )
    ssm[indices] = final_state.to(ssm.dtype)
    return o, ssm


def run_new_path(q, k, v, g, beta, scale, cu_seqlens, ssm_state, indices, has_init):
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule

    ssm = ssm_state.clone()
    zero_mask = ~has_init
    ssm[indices[zero_mask]] = 0
    o, _ = chunk_gated_delta_rule(
        q, k, v, g, beta, scale=scale,
        initial_state=ssm, output_final_state=True, cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )
    return o, ssm


def main():
    device = "cuda"
    dtype = torch.bfloat16

    # Configs matching Qwen3.5-9B GDN layers: H=8 or 16, K=V=128
    configs = [
        # (N, T_per_seq, H, V, K, max_blocks)
        (1, 16,  8, 128, 128, 32),
        (4, 128, 8, 128, 128, 32),
        (8, 256, 16, 128, 128, 32),
        (16, 128, 8, 128, 128, 64),
    ]

    print("=" * 70)
    print("Test 1: Precision (OLD gather/scatter vs NEW in-place)")
    print("=" * 70)

    all_pass = True
    for cfg in configs:
        N, T, H, V, K, mb = cfg
        for seed in [42, 123]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            q, k, v, g, beta, scale, cu_seqlens, ssm, indices, has_init = setup_inputs(
                N, T, H, V, K, mb, device, dtype, seed
            )

            # Warmup both paths (eliminates autotune variance)
            run_old_path(q, k, v, g, beta, scale, cu_seqlens, ssm, indices, has_init)
            torch.cuda.synchronize()
            run_new_path(q, k, v, g, beta, scale, cu_seqlens, ssm, indices, has_init)
            torch.cuda.synchronize()

            # Now measure
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            q2, k2, v2, g2, beta2, scale2, cu2, ssm2, indices2, has_init2 = setup_inputs(
                N, T, H, V, K, mb, device, dtype, seed
            )

            o_old, ssm_old = run_old_path(q2, k2, v2, g2, beta2, scale2, cu2,
                                          ssm2, indices2, has_init2)
            o_new, ssm_new = run_new_path(q2, k2, v2, g2, beta2, scale2, cu2,
                                          ssm2, indices2, has_init2)

            ssm_diff = (ssm_old.float() - ssm_new.float()).abs().max().item()
            o_diff = (o_old.float() - o_new.float()).abs().max().item()

            # ssm_state MUST be exact; o tolerates numerical variance
            ssm_ok = ssm_diff < 1e-5
            o_ok = o_diff < 1e-2  # bfloat16 tolerance

            status = "PASS" if (ssm_ok and o_ok) else "FAIL"
            print(f"  N={N},T={T},H={H},V={V},K={K},seed={seed}: {status}"
                  f"  ssm_diff={ssm_diff:.2e}  o_diff={o_diff:.6f}")

            if not (ssm_ok and o_ok):
                print(f"    ❌ ssm_ok={ssm_ok} o_ok={o_ok}")
                all_pass = False

    if all_pass:
        print("\n  All precision tests PASSED ✓")
    else:
        print("\n  Some tests FAILED ✗")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
