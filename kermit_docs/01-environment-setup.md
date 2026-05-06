# vLLM 0.20.0 环境配置

## 环境信息

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti (16GB) |
| GPU Compute Capability | SM 8.9 (Ada Lovelace) |
| Driver | 560.28.03 |
| OS | Linux 6.8.0-60-generic |
| Conda 环境 | vllm-20 |
| Python | 3.10.0 |
| vLLM | 0.20.0+cu126 |
| PyTorch | 2.11.0+cu126 |
| FlashAttention | v2 |
| FlashInfer | 0.6.8.post1 |

## 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n vllm-20 python=3.10 -y
conda activate vllm-20

# 2. 安装 PyTorch (CUDA 12.6)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3. 安装 vllm (editable, 从源码)
pip install -e /path/to/vllm

# 4. 修复 torchvision 版本匹配
pip install --upgrade torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 模型

本地路径 `.huggingface/` 下已有：

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B

## 注意事项

### ninja 不在 PATH

EngineCore 是独立子进程，flashinfer JIT 编译需要 `ninja`。若遇到 `FileNotFoundError: ninja`：

```bash
# 方案 1: 启动前设置环境变量
export PATH="/home/kermit/.conda/envs/vllm-20/bin:$PATH"
export CMAKE_MAKE_PROGRAM=/home/kermit/.conda/envs/vllm-20/bin/ninja

# 方案 2: 在脚本里设置
import os
os.environ["PATH"] = "/home/kermit/.conda/envs/vllm-20/bin:" + os.environ.get("PATH", "")
os.environ["CMAKE_MAKE_PROGRAM"] = "/home/kermit/.conda/envs/vllm-20/bin/ninja"
```

### torchvision 版本冲突

若安装 vllm 后 import 报 `RuntimeError: operator torchvision::nms does not exist`，需重装匹配版本：

```bash
pip install --upgrade torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Qwen3 thinking 模式

Qwen3 默认开启 thinking（内部推理），会导致输出大量重复 thinking token。用 `/no_think` 后缀或 chat template 关闭。

### Raw prompt deprecation warning

vLLM 0.20 提示 `Passing raw prompts to InputProcessor is deprecated`，建议改用 `Renderer.render_cmpl()` 或 `Renderer.render_chat()`。
