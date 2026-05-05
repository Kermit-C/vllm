"""Run Qwen3-0.6B with FP8 dynamic quantization via vLLM AsyncLLMEngine."""

import os
import asyncio

# FP8 KV cache triggers flashinfer backend which needs ninja for JIT.
# Point cmake/ninja to the conda env binary so the EngineCore subprocess
# can find it.
_conda_bin = "/home/kermit/.conda/envs/vllm-20/bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["CMAKE_MAKE_PROGRAM"] = os.path.join(_conda_bin, "ninja")

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


async def main():
    engine_args = AsyncEngineArgs(
        model=".huggingface/Qwen3.5-9B",
        quantization="fp8",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,
    )

    print("Initializing FP8 engine ...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=128,
    )

    prompt = "Explain quantum computing in 2-3 sentences."
    request_id = "req-fp8-001"

    print(f"Prompt: {prompt}\n")
    print("--- Response ---")

    full_text = ""
    async for output in engine.generate(prompt, params, request_id):
        for comp in output.outputs:
            new_text = comp.text[len(full_text):]
            print(new_text, end="", flush=True)
            full_text = comp.text

    print("\n\n--- Done ---")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
