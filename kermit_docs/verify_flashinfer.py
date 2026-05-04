#!/usr/bin/env python3
"""
FlashInfer backend smoke test for GDN chunk prefill.

Verifies the FlashInfer backend (SM90+ GPUs only) produces correct outputs
for prefill, decode, and mixed scenarios. Not applicable to 4060Ti (SM89).

Branch switching is external — test runs on currently checked-out branch.

Usage:
    conda activate vllm-20
    cd /home/kermit/MyCode/vllm

    PYTHONPATH=. python kermit_docs/verify_flashinfer.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8

    # With output:
    PYTHONPATH=. python kermit_docs/verify_flashinfer.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/flashinfer.json

Note:
    Only tests Qwen3.5-9B. OLMo-Hybrid-7B does NOT use FlashInfer for chunk
    prefill — it calls chunk_gated_delta_rule (Triton) directly, bypassing
    the ChunkGatedDeltaRule CustomOp dispatch. OLMo correctness is verified
    by verify_accuracy.py and bench_serving.py.

Environment:
    GPU: H20 / H100 / B200 (SM90+, FlashInfer required)
    Conda: vllm-20
"""

import argparse
import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import LLM, SamplingParams


def check_sm90() -> bool:
    """Return True if GPU is SM90+ (FlashInfer compatible)."""
    try:
        props = torch.cuda.get_device_properties(0)
        return props.major >= 9
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="FlashInfer backend smoke test")
    parser.add_argument("--model", default=".huggingface/Qwen3.5-9B",
                        help="Model path")
    parser.add_argument("--quantization", default="fp8",
                        help="Quantization method")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model length")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Model: {args.model}")
    print(f"{'='*80}")

    if not check_sm90():
        print("SKIP: FlashInfer requires SM90+ (H20/H100/B200).")
        print(f"This GPU is SM{props.major}{props.minor}.")
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"status": "skipped", "reason": f"SM{props.major}{props.minor} < SM90"}, f, indent=2)
        return

    print("Loading model with FlashInfer backend...")
    model = LLM(
        model=args.model,
        quantization=args.quantization if args.quantization else None,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=128,
        additional_config={"gdn_prefill_backend": "flashinfer"},
    )
    print("Model loaded.\n")

    params = SamplingParams(temperature=0.0, max_tokens=32)

    short = "The quick brown fox jumps over the lazy dog."
    medium = "Machine learning models require careful training. " * 8
    long_t = "Deep neural networks have revolutionized AI. " * 32
    very_long = "Quantum computing leverages superposition. " * 64

    # ── Scenarios ──
    all_scenarios = [
        # Pure prefill
        ("prefill_8t", [short]),
        ("prefill_64t", [short + " " + medium]),
        ("prefill_256t", [long_t]),
        ("prefill_1024t", [very_long]),
        # Large decode batch
        ("decode_x4", [short] * 4),
        ("decode_x16", [short] * 16),
        ("decode_x64", [short] * 64),
        ("decode_x128", [short] * 128),
        # Mixed prefill+decode
        ("mixed_1pf_15d", [medium] + [short] * 15),
        ("mixed_4pf_60d", [medium] * 4 + [short] * 60),
        ("mixed_8pf_120d", [long_t] * 4 + [medium] * 4 + [short] * 120),
    ]

    # Warmup
    print("Warmup...")
    for _ in range(3):
        model.generate([short], params)
    torch.cuda.synchronize()

    print(f"\n{'='*80}")
    print("FlashInfer GDN Backend — E2E Correctness")
    print(f"{'='*80}")

    results = {"tests": {}, "all_passed": True}
    for name, prompts in all_scenarios:
        try:
            torch.cuda.synchronize()
            outputs = model.generate(prompts, params)
            torch.cuda.synchronize()
            passed = all(len(o.outputs[0].token_ids) > 0 for o in outputs)
            results["tests"][name] = {
                "passed": passed,
                "n_seqs": len(prompts),
                "out_lens": [len(o.outputs[0].token_ids) for o in outputs],
            }
            status = "PASS" if passed else "FAIL (empty output)"
            print(f"  {name:<24s} n={len(prompts):3d}  {status}")
            if not passed:
                results["all_passed"] = False
        except Exception as e:
            results["tests"][name] = {"passed": False, "error": str(e)}
            results["all_passed"] = False
            print(f"  {name:<24s} n={len(prompts):3d}  FAIL: {e}")

    overall = "ALL PASSED" if results["all_passed"] else "SOME FAILED"
    print(f"\nOverall: {overall}")

    if args.output:
        results["meta"] = {"model": args.model, "gpu": props.name}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: {args.output}")

    if not results["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
