#!/usr/bin/env python3
"""
Online serving throughput benchmark with concurrency-based pressure.

Uses vllm AsyncEngine directly (no HTTP server) for lower overhead and
reliable concurrency control via worker pool.

Usage:
    conda activate vllm-20
    cd /home/kermit/MyCode/vllm

    # H20: Qwen3.5-9B fp8 triton
    PYTHONPATH=. python kermit_docs/bench_serving.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --h20 \
        --output /tmp/serving_feat_qwen_triton.json

    # FlashInfer backend (SM90+ only)
    PYTHONPATH=. python kermit_docs/bench_serving.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --h20 \
        --gdn-prefill-backend flashinfer --output /tmp/serving_feat_qwen_fi.json

    # Compare results:
    python -c "
    import json
    m = json.load(open('/tmp/serving_main.json'))
    f = json.load(open('/tmp/serving_feat.json'))
    for k in m['results']:
        if m['results'][k].get('success') and f['results'][k].get('success'):
            mr = m['results'][k]['request_throughput']
            fr = f['results'][k]['request_throughput']
            pct = (fr - mr) / mr * 100
            print(f'{k}: main={mr:.1f} feat={fr:.1f} Δ={pct:+.1f}%')
    "

Environment:
    GPU: H20 (96GB+) or RTX 4060 Ti 16GB (0.8B model only)
    Conda: vllm-20
"""

import argparse
import asyncio
import json
import os
import random
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import AsyncLLMEngine, EngineArgs, SamplingParams


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# (input_len, max_output_tokens, concurrency)
SCENARIOS = [
    (512,  128, 1),
    (512,  128, 16),
    (512,  128, 128),
    (1024, 128, 1),
    (1024, 128, 16),
    (1024, 128, 128),
    (2048, 256, 16),
    (2048, 256, 128),
    (4096, 256, 16),
    (4096, 256, 128),
]

H20_SCENARIOS = [
    (512,  128, 1),
    (512,  128, 16),
    (512,  128, 32),
    (512,  128, 128),
    (512,  128, 256),
    (1024, 128, 1),
    (1024, 128, 16),
    (1024, 128, 32),
    (1024, 128, 128),
    (1024, 128, 256),
    (2048, 256, 16),
    (2048, 256, 32),
    (2048, 256, 128),
    (2048, 256, 256),
    (4096, 256, 16),
    (4096, 256, 32),
    (4096, 256, 128),
    (4096, 256, 256),
]


def generate_random_token_ids(num_tokens: int) -> list[int]:
    """Generate random token IDs for benchmarking input."""
    return [random.randint(100, 50000) for _ in range(num_tokens)]


async def send_one_request(engine, prompt_token_ids, sampling_params, request_id):
    """Send one request via AsyncEngine, measure per-token timing."""
    t_start = time.perf_counter()
    ttft = None
    token_times = []
    prev_count = 0
    output_tokens = 0

    try:
        async for output in engine.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
            request_id,
        ):
            now = time.perf_counter()
            if output.outputs:
                cur = len(output.outputs[0].token_ids)
                if cur > prev_count:
                    if ttft is None:
                        ttft = now - t_start
                    token_times.append(now)
                    output_tokens = cur
                    prev_count = cur
            if output.finished:
                break
    except Exception as e:
        return {"success": False, "error": str(e)}

    t_end = time.perf_counter()
    total_latency = t_end - t_start
    itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

    return {
        "success": True,
        "ttft_ms": ttft * 1000 if ttft else None,
        "total_latency_ms": total_latency * 1000,
        "output_tokens": output_tokens,
        "tpot_ms": ((token_times[-1] - token_times[0]) / (len(token_times) - 1) * 1000)
                   if len(token_times) >= 2 else None,
        "itl_mean_ms": (sum(itls) / len(itls) * 1000) if itls else None,
    }


async def run_scenario(engine, input_len, max_tokens, concurrency, num_requests):
    """Run one benchmark scenario with fixed-concurrency worker pool."""
    prompt_token_ids = generate_random_token_ids(input_len)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1.0)

    results = []
    queue = asyncio.Queue()
    for i in range(num_requests):
        queue.put_nowait(i)

    async def worker():
        while True:
            try:
                rid = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            r = await send_one_request(
                engine, prompt_token_ids, sampling_params, f"bench_{rid}")
            results.append(r)

    t_start = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    t_end = time.perf_counter()

    wall_time_s = t_end - t_start
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if not successful:
        return {
            "input_len": input_len,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "success": False,
            "num_failed": len(failed),
        }

    ttfts = sorted([r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None])
    tpots = [r["tpot_ms"] for r in successful if r.get("tpot_ms") is not None]
    latencies = sorted([r["total_latency_ms"] for r in successful])
    total_output_tokens = sum(r.get("output_tokens", 0) for r in successful)

    def pct(sl, p):
        if not sl:
            return None
        return round(sl[min(int(len(sl) * p / 100), len(sl) - 1)], 2)

    return {
        "input_len": input_len,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "success": True,
        "num_success": len(successful),
        "num_failed": len(failed),
        "wall_time_s": round(wall_time_s, 2),
        "request_throughput": round(len(successful) / wall_time_s, 2),
        "output_throughput": round(total_output_tokens / wall_time_s, 2),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "ttft_median_ms": pct(ttfts, 50),
        "ttft_p99_ms": pct(ttfts, 99),
        "tpot_mean_ms": round(sum(tpots) / len(tpots), 2) if tpots else None,
        "tpot_p99_ms": pct(sorted(tpots), 99) if tpots else None,
        "latency_mean_ms": round(sum(latencies) / len(latencies), 2),
        "latency_p99_ms": pct(latencies, 99),
    }


async def warmup(engine):
    """Send a few warmup requests."""
    prompt = generate_random_token_ids(128)
    sp = SamplingParams(max_tokens=32, temperature=1.0)
    tasks = [
        send_one_request(engine, prompt, sp, f"warmup_{i}")
        for i in range(4)
    ]
    await asyncio.gather(*tasks)


def ffmt(v, unit=""):
    return f"{v:.1f}{unit}" if v is not None else "N/A"


async def async_main(args):
    scenarios = H20_SCENARIOS if args.h20 else SCENARIOS

    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Eager: {args.eager}")
    print(f"H20 mode: {args.h20}")
    print(f"  gdn-prefill-backend: {args.gdn_prefill_backend}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"{'='*80}")

    print("\nInitializing AsyncEngine...")
    engine_args = EngineArgs(
        model=args.model,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
        additional_config={"gdn_prefill_backend": args.gdn_prefill_backend},
    )
    if not hasattr(engine_args, "enable_log_requests"):
        engine_args.enable_log_requests = False
    if not hasattr(engine_args, "enable_log_stats"):
        engine_args.enable_log_stats = False
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("  Engine ready.\n")

    print("Warming up...")
    await warmup(engine)
    print("  Warmup done.\n")

    results = {}
    for i, (il, mt, cc) in enumerate(scenarios):
        name = f"in{il}_out{mt}_c{cc}"
        num_req = args.num_requests or min(max(cc * 4, 64), 512)
        print(f"[{i+1}/{len(scenarios)}] {name} (n={num_req}) ...")
        t0 = time.perf_counter()
        result = await run_scenario(engine, il, mt, cc, num_req)
        elapsed = time.perf_counter() - t0
        results[name] = result

        if result.get("success"):
            print(f"  {elapsed:.0f}s  "
                  f"rps={ffmt(result.get('request_throughput'))}  "
                  f"ttft={ffmt(result.get('ttft_mean_ms'), 'ms')}  "
                  f"tpot={ffmt(result.get('tpot_mean_ms'), 'ms')}")
        else:
            print(f"  FAILED after {elapsed:.0f}s")

    output = {
        "meta": {
            "model": args.model,
            "quantization": args.quantization,
            "eager": args.eager,
            "h20_mode": args.h20,
            "gdn_prefill_backend": args.gdn_prefill_backend,
            "max_model_len": args.max_model_len,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Serving throughput benchmark (AsyncEngine, concurrency-based)")
    parser.add_argument("--model", default=".huggingface/Qwen3.5-9B")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--h20", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gdn-prefill-backend", default="triton",
                        choices=["flashinfer", "triton"])
    parser.add_argument("--num-requests", type=int, default=0,
                        help="Requests per scenario (0=auto: max(cc*4,64), cap 512)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
