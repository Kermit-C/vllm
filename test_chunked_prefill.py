"""4 decode + 1 prefill mixed step test.

Scenario:
  1. Send 4 short requests → they finish prefill quickly, enter decode
  2. While all 4 are decoding, send 1 long prefill request (~1200 tokens)
  3. Scheduler batches them: 4 decode + 1 prefill chunk in the same step
"""

import os
import asyncio
import time

_conda_bin = "/home/kermit/.conda/envs/vllm-20/bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["CMAKE_MAKE_PROGRAM"] = os.path.join(_conda_bin, "ninja")

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


def make_long_prompt(n_tokens: int = 1200) -> str:
    base = (
        "Quantum computing is a rapidly evolving field that leverages "
        "the principles of quantum mechanics to perform computations. "
    )
    reps = max(1, n_tokens // 20)
    return base * reps


async def generate(engine, prompt, params, request_id):
    full_text = ""
    async for output in engine.generate(prompt, params, request_id):
        for comp in output.outputs:
            new = comp.text[len(full_text):]
            if new:
                print(f"[{request_id}] {new}", end="", flush=True)
                full_text = comp.text
    print(f"\n[{request_id}] --- finished ---")
    return full_text


async def main():
    engine_args = AsyncEngineArgs(
        model=".huggingface/Qwen3.5-9B",
        quantization="fp8",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_batched_tokens=256,
        enforce_eager=True,
    )

    print("Initializing engine ...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 4 prompts that require long answers — guaranteed to still be decoding
    # when the prefill request arrives 2s later.
    short_prompts = [
        "Write a short essay about the history of artificial intelligence.",
        "Explain the differences between TCP and UDP in detail.",
        "Describe the process of photosynthesis step by step.",
        "Summarize the major events of World War II.",
        "你加一等于几"
    ]
    params_decode = SamplingParams(temperature=0.7, max_tokens=256)

    # 1 long prompt — forces chunked prefill
    long_prompt = make_long_prompt(1200)
    params_prefill = SamplingParams(temperature=0.7, max_tokens=32)

    print("\nStrategy: 4 decode reqs (long answers) + 1 prefill req (1200 tok prompt)")
    print("Decode reqs start first, prefill joins after 2s delay\n")

    t0 = time.time()

    # Launch 4 decode requests
    decode_tasks = [
        asyncio.create_task(
            generate(engine, p, params_decode, f"decode-{i}"))
        for i, p in enumerate(short_prompts)
    ]

    # Wait 2s so all 4 reqs finish prefill and are deep into decode phase
    await asyncio.sleep(2.0)

    # Now launch the long prefill — it will be chunked while the other 4 decode
    prefill_task = asyncio.create_task(
        generate(engine, long_prompt, params_prefill, "prefill-0"))

    await asyncio.gather(*decode_tasks, prefill_task)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
