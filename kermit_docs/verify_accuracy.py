#!/usr/bin/env python3
"""
Accuracy verification using lm_eval (gsm8k 5-shot) for GDN models.

Runs lm_eval on the currently checked-out branch via subprocess.
Branch switching is external — run once on main, once on feature, compare.

Usage:
    conda activate vllm-20
    cd /home/kermit/MyCode/vllm

    # On main branch:
    git checkout main
    python kermit_docs/verify_accuracy.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/acc_main.json

    # On feature branch:
    git checkout feature/gdn-prefill-kernal-opt
    python kermit_docs/verify_accuracy.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --output /tmp/acc_feat.json

    # OLMo:
    python kermit_docs/verify_accuracy.py \
        --model .huggingface/OLMo-Hybrid-7B --quantization fp8 --output /tmp/acc_olmo.json

    # Quick smoke test (limit=50):
    python kermit_docs/verify_accuracy.py \
        --model .huggingface/Qwen3.5-9B --quantization fp8 --limit 50 --output /tmp/acc_smoke.json

    # Compare:
    python -c "
    import json
    m=json.load(open('/tmp/acc_main.json'))
    f=json.load(open('/tmp/acc_feat.json'))
    for t in m['tasks']:
        ma = m['tasks'][t]['accuracy']
        fa = f['tasks'][t]['accuracy']
        print(f'{t}: main={ma:.4f} feat={fa:.4f} match={abs(ma-fa)<1e-6}')
    "

    Expected: feature branch accuracy identical to main (kernel is bit-exact).

Environment:
    GPU: any CUDA GPU with sufficient VRAM
    Conda: vllm-20
    Requires: lm_eval (pip install lm_eval)
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_lm_eval(model: str, tasks: str, quantization: str,
                num_fewshot: int, batch_size: str,
                max_model_len: int, limit: int | None) -> dict:
    """Run lm_eval via subprocess, return parsed accuracy."""
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    model_args = f"pretrained={model},dtype=auto,gpu_memory_utilization=0.85,max_model_len={max_model_len},quantization={quantization},max_num_seqs=16,trust_remote_code=True"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", batch_size,
        "--num_fewshot", str(num_fewshot),
        "--output_path", "/tmp",
        "--log_samples",
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"  Running: lm_eval --model vllm --tasks {tasks} ...")
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=3600,
        cwd=REPO_DIR,
    )

    combined = result.stdout + "\n" + result.stderr
    parsed = {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "tasks": {},
    }

    for task_name in tasks.split(","):
        task_name = task_name.strip()
        # Standard lm_eval table format: "| gsm8k | 5 | ... | 0.xxxx | ... |"
        pattern = rf"\|\s*{re.escape(task_name)}\s*\|\s*\d+\s*\|.*?\|\s*([\d.]+)\s*\|"
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            parsed["tasks"][task_name] = {"accuracy": float(match.group(1))}
        else:
            # Try alternate pattern
            pat2 = rf"{re.escape(task_name)}.*?exact_match[^\n]*?([\d.]+)"
            match2 = re.search(pat2, combined, re.IGNORECASE)
            if match2:
                parsed["tasks"][task_name] = {"accuracy": float(match2.group(1))}
            else:
                parsed["tasks"][task_name] = {"accuracy": None, "parse_error": True}

    for task_name, td in parsed["tasks"].items():
        if td.get("parse_error"):
            print(f"    WARNING: Could not parse {task_name}. Tail:")
            for line in combined.splitlines()[-12:]:
                print(f"      | {line}")

    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy verification via lm_eval for GDN models")
    parser.add_argument("--model", default=".huggingface/Qwen3.5-9B",
                        help="Model path")
    parser.add_argument("--quantization", default="fp8",
                        help="Quantization (default: fp8)")
    parser.add_argument("--tasks", default="gsm8k",
                        help="lm_eval tasks (default: gsm8k)")
    parser.add_argument("--num-fewshot", type=int, default=5,
                        help="Few-shot examples (default: 5)")
    parser.add_argument("--batch-size", default="auto",
                        help="Batch size (default: auto)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model length (default: 4096)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit eval examples (None = full dataset)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Quantization: {args.quantization}")
    print(f"Tasks: {args.tasks} ({args.num_fewshot}-shot)")
    print(f"Limit: {args.limit or 'full dataset'}")
    print(f"{'='*80}")

    results = run_lm_eval(
        model=args.model,
        tasks=args.tasks,
        quantization=args.quantization,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        limit=args.limit,
    )

    output = {
        "meta": {
            "model": args.model,
            "quantization": args.quantization,
            "tasks": args.tasks,
            "num_fewshot": args.num_fewshot,
            "limit": args.limit,
        },
        **results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {args.output}")

    for task_name, td in results.get("tasks", {}).items():
        acc = td.get("accuracy")
        status = f"accuracy={acc:.4f}" if acc is not None else "parse failed"
        print(f"  {task_name}: {status}")


if __name__ == "__main__":
    main()
