#!/bin/bash
# scripts/setup_env.sh
# =====================
# 完整环境搭建脚本（基于真实组件，非 mock）
# 
# 前置条件：Ubuntu 22.04 + Docker + NVIDIA GPU（训练用）
# 纯评估模式（无 GPU）也可运行，用 GPT-4o 作为策略
#
# 用法：bash scripts/setup_env.sh

set -euo pipefail

echo "=== BixBench RL Agent 环境搭建 ==="

# ── 1. 系统依赖 ─────────────────────────────────────────────────
echo "[1/9] 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    git curl wget build-essential docker.io \
    python3.12 python3.12-dev python3-pip

# UV（推荐的包管理器）
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# ── 2. Python 虚拟环境 ──────────────────────────────────────────
echo "[2/9] 创建虚拟环境..."
uv venv --clear --python 3.12 .venv
source .venv/bin/activate

# ── 3. 克隆所有真实仓库 ─────────────────────────────────────────
echo "[3/9] 克隆组件仓库..."

# FutureHouse Aviary（真实 Notebook 环境）
if [ ! -d "aviary" ]; then
    git clone https://github.com/Future-House/aviary.git
fi

# FutureHouse ldp（agent 框架，已从 aviary 拆分）
if [ ! -d "ldp" ]; then
    git clone https://github.com/Future-House/ldp.git
fi

# FutureHouse data-analysis-crow（数据分析 crow 环境，Nemotron 训练用）
if [ ! -d "data-analysis-crow" ]; then
    git clone https://github.com/Future-House/data-analysis-crow.git
fi

# FutureHouse BixBench（数据集 + 评估，可选）
if [ ! -d "BixBench" ]; then
    git clone https://github.com/Future-House/BixBench.git
fi

# verifiers（GRPO 训练框架，alexandonian/verifiers fork）
if [ ! -d "verifiers" ]; then
    git clone https://github.com/alexandonian/verifiers.git
fi

# NVIDIA NeMo Gym（RL 环境构建 + rollout 扩展）
if [ ! -d "Gym" ]; then
    git clone https://github.com/NVIDIA-NeMo/Gym.git
fi

# NVIDIA NeMo RL（GRPO + 步级训练）
if [ ! -d "RL" ]; then
    git clone https://github.com/NVIDIA-NeMo/RL.git
fi

# ── 4. 安装 Aviary（带 Notebook 支持）─────────────────────────
echo "[4/9] 安装 FutureHouse Aviary..."
cd aviary
# 注意：必须从源码安装，PyPI 上的 'aviary' 是 NASA 的飞机设计工具
uv pip install -e ".[notebook]"
uv pip install -e "packages/notebook"
cd ..

echo "[?/9] 安装 FutureHouse ldp..."
cd ldp
uv pip install -e .
cd ..

# ── 5. 安装 data-analysis-crow ──────────────────────────────────
echo "[5/9] 安装 FutureHouse data-analysis-crow..."
cd data-analysis-crow
uv pip install -e .
cd ..

# ── 6. 安装 verifiers（GRPO 框架）──────────────────────────────
echo "[6/9] 安装 verifiers..."
cd verifiers
uv pip install -e ".[all]"
cd ..

# ── 7. 安装 BixBench（可选，用于 BixBench 评估）───────────────
echo "[7/9] 安装 BixBench（可选）..."
cd BixBench
uv pip install -e .
cd ..

# ── 8. 安装 NeMo RL ─────────────────────────────────────────────
echo "[8/9] 安装 NeMo RL（训练框架）..."
cd RL

git fetch --all
git checkout e5a729cc438ea71bafa7138204f861196598a9b2
git submodule update --init --recursive

source ../.venv/bin/activate

# NeMo RL uses uv dependency-groups (not extras) for dev deps
uv sync --group dev --group build --active

# 按 NeMo RL 自己的 pyproject.toml 约束安装 torch（它用 cu129 index）
uv pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu129

cd ..

# ── 9. 安装本项目依赖 ────────────────────────────────────────────
echo "[9/9] 安装本项目..."
uv pip install -e ".[dev,train]"

# ── Docker 镜像（BixBench 生物信息学执行环境）──────────────────
echo ""
echo "=== 拉取 BixBench Docker 环境 ==="
echo "（此镜像包含 Python + R + 完整生物信息学包，约 8GB）"
# data-analysis-crow 使用的镜像
docker pull futurehouse/bixbench:aviary-notebook-env
# 旧版 BixBench 评估镜像（向后兼容）
docker pull futurehouse/bixbench-env:v1.0

# ── 验证安装 ────────────────────────────────────────────────────
echo ""
echo "=== 验证安装 ==="
python -c "
from aviary.envs.notebook.env import NBEnvironment
from ldp.agent import ReActAgent
print('✓ aviary.envs.notebook.NBEnvironment OK')
print('✓ ldp.agent.ReActAgent OK')
"

python -c "
from fhda.data_analysis_env import DataAnalysisEnv
print('✓ fhda.DataAnalysisEnv OK')
"

python -c "
import verifiers as vf
print(f'✓ verifiers {vf.__version__} OK')
"

python -c "
from datasets import load_dataset
print('✓ HuggingFace datasets OK')
print('  To load Nemotron dataset: load_dataset(\"nvidia/Nemotron-RL-bixbench_hypothesis\")')
"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "下一步："
echo "  1. 设置 API Key:    export OPENAI_API_KEY=sk-..."
echo "  2. HuggingFace 认证: huggingface-cli login"
echo "  3. 评估 (no GPU):   python scripts/train_grpo_hypothesis.py \\"
echo "                            --config configs/grpo_hypothesis.yaml"
echo "  4. GPU 训练:"
echo "       # 启动 vLLM 推理服务"
echo "       vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\"
echo "           --port 10240 --max-model-len 32768 --enable-auto-tool-choice"
echo "       # 启动 GRPO 训练（verifiers + accelerate）"
echo "       CUDA_VISIBLE_DEVICES=2,3 accelerate launch \\"
echo "           --config-file configs/zero3.yaml \\"
echo "           scripts/train_grpo_hypothesis.py \\"
echo "           --config configs/grpo_hypothesis.yaml \\"
echo "           --use-verifiers-trainer"
echo ""
