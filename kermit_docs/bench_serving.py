#!/usr/bin/env python3
"""
Online serving throughput benchmark using vllm bench serve.

Starts a vllm server, then runs vllm bench serve against it to measure TTFT, TPOT,
ITL, and throughput under various load conditions.

Usage:
    conda activate vllm-20
    cd /home/kermit/MyCode/vllm

    # Qwen3.5-0.8B bf16 (CUDAGraph, fits on 16GB):
    git checkout main
    python kermit_docs/bench_serving.py \
        --model .huggingface/Qwen3.5-0.8B --output /tmp/serving_main.json
    git checkout feature/gdn-prefill-kernal-opt
    python kermit_docs/bench_serving.py \
        --model .huggingface/Qwen3.5-0.8B --output /tmp/serving_feat.json

    # Qwen3.5-9B fp8 (H20 or 4060Ti with --eager):
    python kermit_docs/bench_serving.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --eager --output /tmp/serving.json

    # Compare:
    python -c "
    import json
    m=json.load(open('/tmp/serving_main.json'))
    f=json.load(open('/tmp/serving_feat.json'))
    for k in m['results']:
        if m['results'][k]['success'] and f['results'][k]['success']:
            dm = m['results'][k]['request_throughput'] - f['results'][k]['request_throughput']
            pct = dm / m['results'][k]['request_throughput'] * 100
            print(f'{k}: main={m[\"results\"][k][\"request_throughput\"]:.1f}rps feat={f[\"results\"][k][\"request_throughput\"]:.1f}rps Δ={dm:+.1f} ({pct:+.1f}%)')
    "

Environment:
    GPU: RTX 4060 Ti 16GB (qwen0.8b bf16 only) or H20
    Conda: vllm-20
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default scenarios — (input_len, output_len, request_rate, num_prompts)
SCENARIOS = [
    (512, 128, "inf", 1024),
    (1024, 128, "inf", 1024),
    (512, 128, "2", 1024),
    (1024, 128, "2", 1024),
]

H20_SCENARIOS = [
    (512, 128, "inf", 1024),
    (1024, 128, "inf", 1024),
    (2048, 256, "inf", 1024),
    (4096, 256, "inf", 1024),
    (512, 128, "2", 1024),
    (1024, 128, "2", 1024),
    (2048, 256, "2", 1024),
    (512, 128, "1", 1024),
    (1024, 128, "1", 1024),
]


def wait_for_server(host: str, port: int, timeout: int = 300) -> bool:
    """Poll /health until server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://{host}:{port}/health")
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_bench(model: str, input_len: int, output_len: int,
              request_rate: str, num_prompts: int,
              host: str = "127.0.0.1", port: int = 8000) -> dict | None:
    """Run one vllm bench serve scenario via subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR

    code = (
        "import sys\n"
        f"sys.path.insert(0, '{REPO_DIR}')\n"
        "from vllm.entrypoints.cli.main import main\n"
        "sys.argv = [\n"
        "    'vllm', 'bench', 'serve',\n"
        f"    '--model', '{model}',\n"
        f"    '--host', '{host}',\n"
        f"    '--port', str({port}),\n"
        "    '--endpoint', '/v1/completions',\n"
        "    '--dataset-name', 'random',\n"
        f"    '--random-input-len', str({input_len}),\n"
        f"    '--random-output-len', str({output_len}),\n"
        f"    '--request-rate', '{request_rate}',\n"
        f"    '--num-prompts', str({num_prompts}),\n"
        "]\n"
        "main()\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True, timeout=600,
            cwd=REPO_DIR,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}

    combined = result.stdout + "\n" + result.stderr

    patterns = {
        "request_throughput": r"Request throughput \(req/s\):\s*([\d.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s*([\d.]+)",
        "mean_ttft": r"Mean TTFT \(ms\):\s*([\d.]+)",
        "median_ttft": r"Median TTFT \(ms\):\s*([\d.]+)",
        "p99_ttft": r"P99 TTFT \(ms\):\s*([\d.]+)",
        "mean_tpot": r"Mean TPOT \(ms\):\s*([\d.]+)",
        "median_tpot": r"Median TPOT \(ms\):\s*([\d.]+)",
        "p99_tpot": r"P99 TPOT \(ms\):\s*([\d.]+)",
        "mean_itl": r"Mean ITL \(ms\):\s*([\d.]+)",
        "median_itl": r"Median ITL \(ms\):\s*([\d.]+)",
        "p99_itl": r"P99 ITL \(ms\):\s*([\d.]+)",
    }

    parsed = {
        "input_len": input_len,
        "output_len": output_len,
        "request_rate": request_rate,
        "num_prompts": num_prompts,
        "success": result.returncode == 0,
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, combined)
        parsed[key] = float(match.group(1)) if match else None

    return parsed


def build_serve_cmd(model, host, port, max_model_len, gpu_memory_utilization,
                    eager, quantization):
    """Build the Python command string to start the vllm server."""
    args_str = (
        f"'vllm', 'serve', '{model}', "
        f"'--host', '{host}', "
        f"'--port', str({port}), "
        f"'--max-model-len', str({max_model_len}), "
        f"'--gpu-memory-utilization', str({gpu_memory_utilization}), "
    )
    if eager:
        args_str += "'--enforce-eager', "
    if quantization:
        args_str += f"'--quantization', '{quantization}', "

    code = (
        "import sys\n"
        f"sys.path.insert(0, '{REPO_DIR}')\n"
        "from vllm.entrypoints.cli.main import main\n"
        f"sys.argv = [{args_str}]\n"
        "main()\n"
    )
    return [sys.executable, "-c", code]


def main():
    parser = argparse.ArgumentParser(
        description="Serving throughput benchmark for GDN models")
    parser.add_argument("--model", default=".huggingface/Qwen3.5-9B",
                        help="Model path")
    parser.add_argument("--quantization", default=None,
                        help="Quantization (fp8, etc.)")
    parser.add_argument("--eager", action="store_true",
                        help="Use enforce_eager mode")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--h20", action="store_true",
                        help="Use H20 extended scenarios")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=0,
                        help="Server port (0 = auto-pick free port)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model length")
    args = parser.parse_args()

    # Auto-pick a free port if not specified
    if args.port == 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            args.port = s.getsockname()[1]

    scenarios = H20_SCENARIOS if args.h20 else SCENARIOS

    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Eager: {args.eager}")
    print(f"H20 mode: {args.h20}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"{'='*80}")

    # ── Start vllm server ──
    print("\nStarting vllm server...")
    serve_cmd = build_serve_cmd(
        args.model, args.host, args.port, args.max_model_len,
        args.gpu_memory_utilization, args.eager, args.quantization,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    server_proc = subprocess.Popen(
        serve_cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=REPO_DIR,
    )

    # Wait for server to be ready
    print(f"  Waiting for server to be ready on http://{args.host}:{args.port} ...")
    if not wait_for_server(args.host, args.port):
        print("  ERROR: Server failed to start within timeout")
        server_proc.kill()
        server_proc.wait()
        try:
            out, err = server_proc.communicate(timeout=5)
            print(f"  Server stdout tail: {out[-500:] if out else 'none'}")
            print(f"  Server stderr tail: {err[-500:] if err else 'none'}")
        except Exception:
            pass
        sys.exit(1)
    print("  Server is ready.\n")

    # ── Run benchmarks ──
    results = {}
    try:
        for i, (il, ol, rr, np_) in enumerate(scenarios):
            name = f"in{il}_out{ol}_r{rr}"
            print(f"[{i+1}/{len(scenarios)}] {name} ...")
            t0 = time.perf_counter()
            parsed = run_bench(args.model, il, ol, rr, np_, args.host, args.port)
            elapsed = time.perf_counter() - t0
            if parsed:
                results[name] = parsed
                thr = parsed.get("request_throughput", "N/A")
                ttft = parsed.get("mean_ttft", "N/A")
                tpot = parsed.get("mean_tpot", "N/A")
                print(f"  {elapsed:.0f}s  throughput={thr} req/s  "
                      f"TTFT={ttft}ms  TPOT={tpot}ms")
            else:
                print(f"  FAILED after {elapsed:.0f}s")
                results[name] = {"success": False}
    finally:
        # ── Stop server ──
        print("\nStopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server stopped.")

    output = {
        "meta": {
            "model": args.model,
            "quantization": args.quantization,
            "eager": args.eager,
            "h20_mode": args.h20,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
