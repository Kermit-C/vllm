import torch
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule


def _make_inputs(N, T_per_seq, H, K, V, max_blocks, device='cuda', seed=42):
    """Create controlled test inputs that avoid numerical overflow."""
    torch.manual_seed(seed)
    total_T = N * T_per_seq
    dtype = torch.bfloat16

    q = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(1, total_T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(1, total_T, H, V, device=device, dtype=dtype) * 0.1
    # g is in log space — keep values small to avoid exp overflow
    g = torch.randn(1, total_T, H, device=device, dtype=torch.float32) * 0.1
    # beta is sigmoid-activated — already in [0,1]
    beta = torch.rand(1, total_T, H, device=device, dtype=torch.float32).sigmoid()
    cu_seqlens = torch.arange(0, total_T + 1, T_per_seq, device=device, dtype=torch.int32)
    ssm_state = torch.randn(max_blocks, H, V, K, device=device, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, cu_seqlens, ssm_state


def test_chunk_ssm_state_indices_correctness():
    """Verify in-place (ssm_state_indices) produces identical results to gather/scatter."""
    N, T_per_seq, H, K, V = 4, 256, 8, 128, 128
    max_blocks = 16
    q, k, v, g, beta, cu_seqlens, ssm_state = _make_inputs(
        N, T_per_seq, H, K, V, max_blocks)

    # Non-identity mapping to test pointer arithmetic thoroughly
    indices = torch.tensor([5, 2, 8, 1], device='cuda', dtype=torch.int32)

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
    ssm_pre = ssm_ip.clone()
    o_ip, final_ip = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=ssm_ip,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )

    # final_ip shares memory with ssm_ip (in-place write)
    assert final_ip.data_ptr() == ssm_ip.data_ptr(), (
        "final_state should share memory with ssm_state (in-place)"
    )
    # ssm_ip must have been modified at the indexed positions
    assert not torch.equal(ssm_ip[indices], ssm_pre[indices]), (
        "ssm_state indexed blocks should be modified in-place by the kernel"
    )

    # Numeric verification
    torch.testing.assert_close(o_ref, o_ip, atol=1e-2, rtol=1e-2,
        msg="Output mismatch between gather/scatter and in-place")
    torch.testing.assert_close(ssm_ref, ssm_ip, atol=1e-2, rtol=1e-2,
        msg="ssm_state mismatch between gather/scatter and in-place")
    print("✓ test_chunk_ssm_state_indices_correctness PASSED")


def test_chunk_null_block_id():
    """Verify NULL_BLOCK_ID entries are no-ops (no crash, correct output)."""
    N, T_per_seq, H, K, V = 3, 256, 8, 128, 128
    max_blocks = 16
    q, k, v, g, beta, cu_seqlens, ssm_state = _make_inputs(
        N, T_per_seq, H, K, V, max_blocks, seed=123)

    # Index 0 → NULL_BLOCK_ID (no state), others → valid blocks
    indices = torch.tensor([0, 5, 3], device='cuda', dtype=torch.int32)

    # -- Reference: gather/scatter with proper zeroing --
    ssm_ref = ssm_state.clone()
    ssm_ref[0] = 0  # zero NULL block in cache (same as caller)
    initial_state_ref = ssm_ref[indices].contiguous().clone()
    o_ref, final_ref = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=initial_state_ref,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ssm_ref[5] = final_ref[1].to(ssm_ref.dtype)
    ssm_ref[3] = final_ref[2].to(ssm_ref.dtype)

    # -- Test: in-place path --
    ssm_ip = ssm_state.clone()
    ssm_ip[0] = 0  # zero cache entry for seq 0 (NULL_BLOCK_ID)
    o_ip, final_ip = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=ssm_ip,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
    )

    # Block 0 should still be all zeros (not written)
    assert (ssm_ip[0] == 0).all(), "NULL_BLOCK_ID cache entry should remain zero"
    # Block 5 should have been updated
    assert not torch.equal(ssm_ip[5], ssm_state[5]), (
        "Valid cache block 5 should have been updated")

    torch.testing.assert_close(o_ref, o_ip, atol=1e-2, rtol=1e-2,
        msg="Output mismatch with NULL_BLOCK_ID handling")
    torch.testing.assert_close(ssm_ref, ssm_ip, atol=1e-2, rtol=1e-2,
        msg="ssm_state mismatch with NULL_BLOCK_ID handling")
    print("✓ test_chunk_null_block_id PASSED")


def test_chunk_backward_compat():
    """Without ssm_state_indices, behavior must be identical to before."""
    B, T, H, K, V = 2, 256, 8, 128, 128
    device = 'cuda'
    dtype = torch.bfloat16

    q = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1
    g = torch.randn(B, T, H, device=device, dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, H, device=device, dtype=torch.float32).sigmoid()
    initial_state = torch.randn(B, H, V, K, device=device, dtype=torch.float32) * 0.1

    o, final_state = chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    assert o.shape == (B, T, H, V)
    assert final_state.shape == (B, H, V, K)
    print("✓ test_chunk_backward_compat PASSED")


if __name__ == "__main__":
    test_chunk_backward_compat()
    test_chunk_ssm_state_indices_correctness()
    test_chunk_null_block_id()
    print("\n✓ ALL TESTS PASSED")
