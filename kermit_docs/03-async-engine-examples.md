# vLLM AsyncLLMEngine 使用示例

## BF16 默认推理

脚本: `test_async_engine.py`

```python
"""Run Qwen3-0.6B via vLLM AsyncLLMEngine (BF16 default)."""
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


async def main():
    engine_args = AsyncEngineArgs(
        model=".huggingface/Qwen3-0.6B",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
    prompt = "Explain quantum computing in 2-3 sentences. /no_think"

    full_text = ""
    async for output in engine.generate(prompt, params, "req-001"):
        for comp in output.outputs:
            new_text = comp.text[len(full_text):]
            print(new_text, end="", flush=True)
            full_text = comp.text

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

## FP8 量化推理

脚本: `test_fp8_engine.py`

```python
"""Run Qwen3-0.6B with FP8 dynamic quantization via vLLM AsyncLLMEngine."""
import os
import asyncio

# FP8 KV cache triggers flashinfer JIT, needs ninja in subprocess PATH
_conda_bin = "/home/kermit/.conda/envs/vllm-20/bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["CMAKE_MAKE_PROGRAM"] = os.path.join(_conda_bin, "ninja")

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


async def main():
    engine_args = AsyncEngineArgs(
        model=".huggingface/Qwen3-0.6B",
        quantization="fp8",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
    prompt = "Explain quantum computing in 2-3 sentences."

    full_text = ""
    async for output in engine.generate(prompt, params, "req-fp8-001"):
        for comp in output.outputs:
            new_text = comp.text[len(full_text):]
            print(new_text, end="", flush=True)
            full_text = comp.text

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

## 运行方式

```bash
# 激活环境
conda activate vllm-20

# BF16
python test_async_engine.py

# FP8 (需要 ninja 在 PATH)
PATH="/home/kermit/.conda/envs/vllm-20/bin:$PATH" python test_fp8_engine.py
```

## API 说明

### AsyncEngineArgs 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `model` | 模型路径或 HuggingFace ID | 必填 |
| `quantization` | 量化方式: fp8, gptq, awq 等 | None |
| `kv_cache_dtype` | KV cache 数据类型: auto, fp8, fp8_e4m3fn | auto |
| `gpu_memory_utilization` | GPU 显存利用率 | 0.9 |
| `max_model_len` | 最大序列长度 | 模型默认 |
| `enforce_eager` | 禁用 torch.compile 和 CUDA Graph | False |
| `tensor_parallel_size` | TP 并行度 | 1 |
| `dtype` | 模型数据类型 | auto (通常 bfloat16) |

### SamplingParams 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `temperature` | 采样温度 | 1.0 |
| `top_p` | nucleus sampling | 1.0 |
| `max_tokens` | 最大生成 token 数 | 16 |
| `stop` | 停止词列表 | None |

### generate() 流式输出

`engine.generate()` 返回 `AsyncGenerator[RequestOutput, None]`。

- `output.outputs[0].text` 是**完整已生成文本**（非增量）
- 增量打印: `new_text = comp.text[len(full_text):]`
- `output.finished` 可判断是否生成完毕
