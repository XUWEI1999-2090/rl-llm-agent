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
echo "[1/8] 安装系统依赖..."
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
echo "[2/8] 创建虚拟环境..."
uv venv --python 3.12 .venv
source .venv/bin/activate

# ── 3. 克隆所有真实仓库 ─────────────────────────────────────────
echo "[3/8] 克隆组件仓库..."

# FutureHouse Aviary（真实 Notebook 环境）
if [ ! -d "aviary" ]; then
    git clone https://github.com/Future-House/aviary.git
fi

# FutureHouse BixBench（数据集 + 评估）
if [ ! -d "BixBench" ]; then
    git clone https://github.com/Future-House/BixBench.git
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
echo "[4/8] 安装 FutureHouse Aviary..."
cd aviary
# 注意：必须从源码安装，PyPI 上的 'aviary' 是 NASA 的飞机设计工具
uv pip install -e "packages/aviary[notebook]"
uv pip install -e "packages/aviary.notebook"
# ldp（aviary 的 agent 框架）
uv pip install -e "packages/ldp"
cd ..

# ── 5. 安装 BixBench ────────────────────────────────────────────
echo "[5/8] 安装 BixBench..."
cd BixBench
uv pip install -e .
# 下载数据集
python -m bixbench.download --output ../data/bixbench/
cd ..

# ── 6. 安装 NeMo Gym ────────────────────────────────────────────
echo "[6/8] 安装 NeMo Gym..."
cd Gym
uv sync --extra dev
cd ..

# ── 7. 安装 NeMo RL ─────────────────────────────────────────────
echo "[7/8] 安装 NeMo RL（训练框架）..."
cd RL
uv sync --extra dev
# NeMo RL 需要 PyTorch + CUDA
uv pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd ..

# ── 8. 安装本项目依赖 ────────────────────────────────────────────
echo "[8/8] 安装本项目..."
uv pip install -e ".[dev]"

# ── Docker 镜像（BixBench 生物信息学执行环境）──────────────────
echo ""
echo "=== 拉取 BixBench Docker 环境 ==="
echo "（此镜像包含 Python + R + 完整生物信息学包，约 8GB）"
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
import bixbench
print(f'✓ BixBench {bixbench.__version__} OK')
"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "下一步："
echo "  1. 设置 API Key:    export OPENAI_API_KEY=sk-..."
echo "  2. 快速评估:        bash scripts/eval_bixbench.sh"
echo "  3. 启动训练:        bash scripts/train_grpo.sh"
echo ""
echo "训练需要 GPU，评估用 GPT-4o 不需要 GPU。"
