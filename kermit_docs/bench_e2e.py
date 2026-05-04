#!/usr/bin/env python3
"""
End-to-end inference latency benchmark for GDN models (Qwen3.5-9B, OLMo-Hybrid-7B).

Measures TTFT, TPOT, and total step time across prefill/decode/mixed scenarios
using vLLM LLM.generate() with CUDAGraph (production path, unless --eager).

Usage:
    conda activate vllm-20
    cd /home/kermit/MyCode/vllm

    # Qwen3.5-9B fp8 (required for 16GB GPUs):
    git checkout main
    PYTHONPATH=. python kermit_docs/bench_e2e.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/e2e_main.json
    git checkout feature/gdn-prefill-kernal-opt
    PYTHONPATH=. python kermit_docs/bench_e2e.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/e2e_feat.json

    # OLMo-Hybrid-7B fp8:
    PYTHONPATH=. python kermit_docs/bench_e2e.py \
        --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --output /tmp/e2e_olmo.json

    # Qwen3.5-0.8B bf16 (no quantization needed):
    PYTHONPATH=. python kermit_docs/bench_e2e.py \
        --model .huggingface/Qwen3.5-0.8B --output /tmp/e2e_qwen0.8b.json

    # With eager mode (for debugging / comparison):
    PYTHONPATH=. python kermit_docs/bench_e2e.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --eager --output /tmp/e2e_eager.json

    # Compare results:
    python -c "
    import json
    m=json.load(open('/tmp/e2e_main.json'))
    f=json.load(open('/tmp/e2e_feat.json'))
    for k,v in m['scenarios'].items():
        dm = v['avg_ms'] - f['scenarios'][k]['avg_ms']
        pct = dm / v['avg_ms'] * 100
        print(f'{k:<24s}  main={v[\"avg_ms\"]:8.1f}ms  feat={f[\"scenarios\"][k][\"avg_ms\"]:8.1f}ms  Δ={dm:+6.1f}ms  ({pct:+.1f}%)')
    "

Environment:
    GPU: RTX 4060 Ti 16GB (fp8 required) or H20
    Conda: vllm-20
"""

import argparse
import json
import os
import time

import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

# ── Prompt templates ─────────────────────────────────────────────────────

SHORT = "The quick brown fox jumps over the lazy dog."
MEDIUM = "Machine learning models require careful training and evaluation. " * 8
LONG = "Deep neural networks have revolutionized artificial intelligence. " * 32
VERY_LONG = "Quantum computing leverages quantum mechanical phenomena. " * 64

# ── Benchmark scenarios ──────────────────────────────────────────────────
# Each: (name, [prompts]) — prompts list defines batch composition

SCENARIOS = [
    # Prefill: single sequence, varying length
    ("prefill_64t", [MEDIUM]),
    ("prefill_128t", [LONG]),
    ("prefill_256t", [LONG * 2]),
    ("prefill_512t", [VERY_LONG]),
    ("prefill_1024t", [VERY_LONG * 2]),
    # Decode: short prompts → decode 1 token; varying batch
    ("decode_x1", [SHORT]),
    ("decode_x16", [SHORT] * 16),
    ("decode_x64", [SHORT] * 64),
    ("decode_x128", [SHORT] * 128),
    # Mixed: some prefill, some decode
    ("mixed_1pf_15d", [MEDIUM] + [SHORT] * 15),
    ("mixed_4pf_60d", [MEDIUM] * 4 + [SHORT] * 60),
    ("mixed_8pf_120d", [LONG] * 4 + [MEDIUM] * 4 + [SHORT] * 120),
]

WARMUP_ROUNDS = 3
BENCH_ROUNDS = 10


def run_benchmark(model: LLM, scenarios: list, warmup: int, rounds: int):
    """Run benchmark scenarios, return results dict and warmup model."""
    params = SamplingParams(temperature=0.0, max_tokens=1)

    # Warmup
    print(f"  Warming up ({warmup} rounds)...")
    for i in range(warmup):
        for name, prompts in scenarios:
            model.generate(prompts, params)
        torch.cuda.synchronize()
        print(f"    warmup {i + 1}/{warmup} done")

    # Benchmark
    results = {}
    for name, prompts in scenarios:
        times = []
        for _ in range(rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(prompts, params)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        results[name] = {
            "n_seqs": len(prompts),
            "avg_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "measurements_ms": [round(t, 2) for t in times],
        }
        print(f"  {name:<24s}  n={len(prompts):3d}  "
              f"avg={results[name]['avg_ms']:8.1f}ms  "
              f"min={results[name]['min_ms']:8.1f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="E2E inference latency benchmark for GDN models")
    parser.add_argument("--model", default=".huggingface/Qwen3.5-9B",
                        help="Model path (default: .huggingface/Qwen3.5-9B)")
    parser.add_argument("--quantization", default=None,
                        help="Quantization method (e.g., fp8). Required for 16GB GPUs with 9B/7B models.")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--eager", action="store_true",
                        help="Use enforce_eager (disable CUDAGraph)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model length (default: 4096)")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Max number of sequences (default: 256)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization (default: 0.85)")
    parser.add_argument("--warmup", type=int, default=WARMUP_ROUNDS,
                        help=f"Warmup rounds (default: {WARMUP_ROUNDS})")
    parser.add_argument("--rounds", type=int, default=BENCH_ROUNDS,
                        help=f"Benchmark rounds per scenario (default: {BENCH_ROUNDS})")
    args = parser.parse_args()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name} ({props.total_memory // 1024**3}GB)")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Eager mode: {args.eager}")
    print(f"Rounds: warmup={args.warmup}, bench={args.rounds}")
    print(f"{'='*80}")

    print("Loading model...")
    model = LLM(
        model=args.model,
        quantization=args.quantization if args.quantization else None,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    print("Model loaded.\n")

    scenarios_results = run_benchmark(model, SCENARIOS, args.warmup, args.rounds)

    results = {
        "meta": {
            "model": args.model,
            "quantization": args.quantization,
            "eager": args.eager,
            "max_model_len": args.max_model_len,
            "gpu": props.name,
            "warmup": args.warmup,
            "rounds": args.rounds,
        },
        "scenarios": scenarios_results,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
