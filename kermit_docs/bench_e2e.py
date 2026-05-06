#!/usr/bin/env python3
"""
End-to-end inference latency benchmark for GDN models (Qwen3.5-9B, OLMo-Hybrid-7B).

Fine-grained control over prefill length and decode length per request.
Uses random token IDs to eliminate prefix caching.

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

    # Qwen3.5-0.8B bf16:
    PYTHONPATH=. python kermit_docs/bench_e2e.py \
        --model .huggingface/Qwen3.5-0.8B --output /tmp/e2e_qwen0.8b.json

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
    GPU: RTX 4060 Ti 16GB (fp8 required for 9B/7B) or H20 96GB
    Conda: vllm-20
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass

import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams


# ── Scenario definitions ──────────────────────────────────────────────────


@dataclass
class RequestSpec:
    """A group of requests with the same prompt_len and max_tokens."""

    prompt_len: int
    max_tokens: int
    count: int


@dataclass
class Scenario:
    name: str
    specs: list[RequestSpec]


SCENARIOS = [
    # ── Prefill: single seq, vary prompt length, decode 1 token (measures TTFT)
    Scenario("prefill_64t",   [RequestSpec(64,   1, 1)]),
    Scenario("prefill_128t",  [RequestSpec(128,  1, 1)]),
    Scenario("prefill_256t",  [RequestSpec(256,  1, 1)]),
    Scenario("prefill_512t",  [RequestSpec(512,  1, 1)]),
    Scenario("prefill_1024t", [RequestSpec(1024, 1, 1)]),
    # ── Decode: short prompt, vary batch size, decode 256 tokens
    Scenario("decode_bs1",   [RequestSpec(32, 256, 1)]),
    Scenario("decode_bs16",  [RequestSpec(32, 256, 16)]),
    Scenario("decode_bs64",  [RequestSpec(32, 256, 64)]),
    Scenario("decode_bs128", [RequestSpec(32, 256, 128)]),
    # ── Mixed: prefill-heavy (long prompt, short decode) + decode-heavy
    #    (short prompt, long decode) in the same batch
    Scenario("mixed_1pf_15d",  [
        RequestSpec(512, 8,   1),
        RequestSpec(32,  256, 15),
    ]),
    Scenario("mixed_4pf_60d",  [
        RequestSpec(256, 16,  4),
        RequestSpec(32,  256, 60),
    ]),
    Scenario("mixed_8pf_120d", [
        RequestSpec(256, 16,  8),
        RequestSpec(32,  256, 120),
    ]),
]

WARMUP_ROUNDS = 5
BENCH_ROUNDS = 100


# ── Prompt generation ─────────────────────────────────────────────────────


def make_token_ids(vocab_size: int, n_tokens: int, seed: int) -> list[int]:
    """Generate exactly n_tokens unique random token IDs.

    Avoids special tokens (low/high ranges) and uses per-call seed
    so every prompt is unique — no prefix caching possible.
    """
    rng = random.Random(seed)
    safe_min, safe_max = 100, vocab_size - 100
    return [rng.randint(safe_min, safe_max) for _ in range(n_tokens)]


def build_scenario_inputs(scenario: Scenario, vocab_size: int, round_idx: int):
    """Build per-request token-ID prompts and SamplingParams."""
    all_prompts: list[list[int]] = []
    all_params: list[SamplingParams] = []

    for spec_idx, spec in enumerate(scenario.specs):
        for req_idx in range(spec.count):
            seed = hash((scenario.name, spec_idx, req_idx, round_idx)) & 0xFFFFFFFF
            tokens = make_token_ids(vocab_size, spec.prompt_len, seed)
            all_prompts.append(tokens)
            all_params.append(SamplingParams(temperature=0.0, max_tokens=spec.max_tokens))

    return all_prompts, all_params


# ── Benchmark runner ──────────────────────────────────────────────────────


def run_benchmark(model: LLM, scenarios: list[Scenario], warmup: int, rounds: int):
    tokenizer = model.get_tokenizer()
    vocab_size = tokenizer.vocab_size

    # Verify token counts
    print("\n  Token count verification (random tokens — exact by construction):")
    for scenario in scenarios:
        parts = []
        for spec in scenario.specs:
            parts.append(f"pf={spec.prompt_len} dec={spec.max_tokens} x{spec.count}")
        print(f"    {scenario.name}: {'  '.join(parts)}")

    # Warmup
    print(f"\n  Warming up ({warmup} rounds)...")
    for i in range(warmup):
        for scenario in scenarios:
            prompts, params = build_scenario_inputs(scenario, vocab_size, i)
            model.generate(prompts, sampling_params=params)
        torch.cuda.synchronize()
        print(f"    warmup {i + 1}/{warmup} done")

    # Benchmark
    results = {}
    for scenario in scenarios:
        total_seqs = sum(s.count for s in scenario.specs)
        total_decode_tokens = sum(s.count * s.max_tokens for s in scenario.specs)

        times = []
        for r in range(rounds):
            prompts, params = build_scenario_inputs(
                scenario, vocab_size, r + warmup,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(prompts, sampling_params=params)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        sorted_times = sorted(times)
        avg_ms = sum(times) / len(times)
        median_ms = sorted_times[len(sorted_times) // 2]
        p10_ms = sorted_times[int(len(sorted_times) * 0.1)]
        p90_ms = sorted_times[int(len(sorted_times) * 0.9)]
        throughput = round(total_decode_tokens / (avg_ms / 1000), 1)

        results[scenario.name] = {
            "specs": [
                {
                    "prompt_len": s.prompt_len,
                    "max_tokens": s.max_tokens,
                    "count": s.count,
                }
                for s in scenario.specs
            ],
            "n_seqs": total_seqs,
            "total_decode_tokens": total_decode_tokens,
            "avg_ms": round(avg_ms, 2),
            "median_ms": round(median_ms, 2),
            "p10_ms": round(p10_ms, 2),
            "p90_ms": round(p90_ms, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "throughput_tok_per_s": throughput,
        }

        desc = ", ".join(
            f"{s.count}x(pf={s.prompt_len},dec={s.max_tokens})"
            for s in scenario.specs
        )
        print(
            f"  {scenario.name:<24s}  [{desc}]  "
            f"median={median_ms:8.1f}ms  "
            f"avg={avg_ms:8.1f}ms  "
            f"tp={throughput:8.1f}tok/s"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="E2E inference latency benchmark for GDN models",
    )
    parser.add_argument(
        "--model",
        default=".huggingface/Qwen3.5-9B",
        help="Model path (default: .huggingface/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Quantization method (e.g., fp8). Required for 16GB GPUs with 9B/7B models.",
    )
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Use enforce_eager (disable CUDAGraph)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length (default: 4096)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Max number of sequences (default: 256)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization (default: 0.85)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ROUNDS,
        help=f"Warmup rounds (default: {WARMUP_ROUNDS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=BENCH_ROUNDS,
        help=f"Benchmark rounds per scenario (default: {BENCH_ROUNDS})",
    )
    args = parser.parse_args()

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print(f"GPU: {props.name} ({props.total_memory // 1024**3}GB)")
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Eager mode: {args.eager}")
    print(f"Rounds: warmup={args.warmup}, bench={args.rounds}")
    print(f"{'=' * 80}")

    print("Loading model...")
    model = LLM(
        model=args.model,
        quantization=args.quantization if args.quantization else None,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        additional_config={"gdn_prefill_backend": "triton"},
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
