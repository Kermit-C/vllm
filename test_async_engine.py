"""Quick smoke test: run Qwen3-0.6B via vLLM AsyncLLMEngine."""

import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


async def main():
    engine_args = AsyncEngineArgs(
        model=".huggingface/Qwen3-0.6B",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    print("Initializing engine ...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Qwen3 has thinking mode; set temperature=0.7 + enable thinking
    # or disable thinking with chat template extra_body
    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=256,
    )

    # Use /no_think prompt to disable Qwen3 thinking for cleaner output
    prompt = (
        "Give me a short introduction to large language models in 2-3 sentences. /no_think"
    )
    request_id = "req-001"

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
