# BixBench RL Agent

Edison Scientific / NeMo Gym + Aviary + BixBench 工作的完整复现项目，
并扩展支持 **Nemotron-RL-bixbench_hypothesis** 假设检验训练流水线。

---

## 项目总览

本项目包含**两条**独立的 RL 训练流水线：

| 流水线 | 数据集 | 环境 | 奖励 | 配置文件 |
|--------|--------|------|------|----------|
| **BixBench MCQ** | FutureHouse/BixBench | ControlledNotebookEnv (aviary) | 终局答案 | `configs/grpo_step.yaml` |
| **Hypothesis** ⭐ | nvidia/Nemotron-RL-bixbench_hypothesis | CrowRLEnv (data-analysis-crow) | 步骤级协议奖励 | `configs/grpo_hypothesis.yaml` |

---

## ⭐ 新增：假设检验训练流水线

### 架构

```
NemotronHypothesisDataset              (src/dataset/nemotron_dataset.py)
  ↓ loads nvidia/Nemotron-RL-bixbench_hypothesis from HuggingFace
  ↓ provides HypothesisSample(hypothesis, answer, capsule_id, ...)

CrowRLEnv / CrowDataset                (src/envs/crow_env.py)
  ↓ wraps DataAnalysisEnv from data-analysis-crow (fhda)
  ↓ mounts capsule data → runs DataAnalysisEnv.step() via aviary tools
  ↓ runs in Docker: futurehouse/bixbench:aviary-notebook-env

AnalysisProtocol + StepRubric          (src/verifiers/)
  ↓ 6-step protocol with per-step weights (sum = 1.0)
  ↓ step verifiers: HypothesisUnderstanding, DataLoading,
  ↓                 ExploratoryAnalysis, StatisticalTesting,
  ↓                 Interpretation, AnswerSubmission
  ↓ StepRubric.score_trajectory(traj) → {step_name: score}

StepLevelGRPOGrouper                   (scripts/train_grpo_hypothesis.py)
  ↓ groups G parallel rollouts by (capsule_id, protocol_step)
  ↓ computes per-step advantage: A_t^i = (r_t^i - mean) / std
  ↓ feeds into NeMo RL / verifiers.GRPOTrainer
```

### 协议步骤与得分

| 步骤 | 名称 | 权重 | 得分标准 |
|------|------|------|----------|
| 1 | hypothesis_understanding | 0.10 | 识别假设变量、测试方向 |
| 2 | data_loading | 0.15 | 读取数据文件，调用 list_workdir |
| 3 | exploratory_analysis | 0.20 | head/describe/shape/missing values |
| 4 | statistical_testing | 0.25 | 正确统计检验 + 代码无错误执行 |
| 5 | interpretation | 0.15 | p 值解读 + 假设接受/拒绝推理 |
| 6 | answer_submission | 0.15 | submit_answer("True"/"False") 正确 |
| | **合计** | **1.00** | |

### Step-level GRPO（Edison 创新 #2 扩展）

标准 GRPO 对整条轨迹归一化奖励：
```
G 条轨迹 → 1 个 advantage per trajectory
```

Step-level GRPO 对每个协议步骤独立归一化：
```
G 条轨迹 × 6 个协议步骤 → 6 × G 个 advantages
```

优势：(1) 梯度信号密度 6×，(2) 每步上下文短，(3) 针对性强化薄弱步骤。

---

## 目录结构

```
bixbench-rl-agent/
├── configs/
│   ├── grpo_step.yaml           # BixBench MCQ 步骤级 GRPO 配置
│   └── grpo_hypothesis.yaml     # ⭐ 假设检验 GRPO 配置（新增）
├── src/
│   ├── dataset/
│   │   └── nemotron_dataset.py  # ⭐ Nemotron 数据集加载器（新增）
│   ├── verifiers/
│   │   ├── protocol.py          # ⭐ AnalysisProtocol（新增）
│   │   ├── step_verifiers.py    # ⭐ 6个步骤验证器（新增）
│   │   └── rubric.py            # ⭐ StepRubric（新增）
│   ├── envs/
│   │   ├── notebook_env.py      # BixBench MCQ 环境（原有）
│   │   └── crow_env.py          # ⭐ data-analysis-crow 环境（新增）
│   └── agents/
│       └── notebook_agent.py    # ldp.ReActAgent（原有）
├── scripts/
│   ├── setup_env.sh             # 环境搭建（已更新）
│   ├── eval_bixbench.sh         # BixBench 评估
│   ├── train_grpo.py            # BixBench MCQ 训练（原有）
│   └── train_grpo_hypothesis.py # ⭐ 假设检验训练（新增）
└── tests/
    ├── test_api_compatibility.py
    └── run_single_capsule.py
```

---

## 快速开始

### 1. 搭建环境

```bash
git clone <this-repo>
cd bixbench-rl-agent
bash scripts/setup_env.sh
```

`setup_env.sh` 安装：
- `github.com/Future-House/aviary` — Notebook 环境
- `github.com/Future-House/data-analysis-crow` — ⭐ 数据分析 Crow 环境
- `github.com/alexandonian/verifiers` — ⭐ GRPO 训练框架
- `github.com/Future-House/BixBench` — BixBench 评估
- `github.com/NVIDIA-NeMo/RL` — NeMo RL
- `docker pull futurehouse/bixbench:aviary-notebook-env` — 执行环境

### 2. 认证

```bash
export OPENAI_API_KEY=sk-...          # 或 vLLM 服务的 key
huggingface-cli login                  # 下载 Nemotron 数据集需要认证
```

### 3. 调试运行（无 GPU）

```bash
python scripts/train_grpo_hypothesis.py --config configs/grpo_hypothesis.yaml
```

### 4. 完整 GPU 训练

```bash
# 终端 1: 启动 vLLM 推理服务
CUDA_VISIBLE_DEVICES=0,1 vf-vllm \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --tensor-parallel-size 2

# 终端 2: 启动 GRPO 训练
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num-processes 2 \
    --config-file configs/zero3.yaml \
    scripts/train_grpo_hypothesis.py \
    --config configs/grpo_hypothesis.yaml \
    --use-verifiers-trainer
```

### 5. NeMo RL 模式（8×H100）

```bash
ng_run \
    +config_paths=[configs/grpo_hypothesis.yaml] \
    trainer.grpo_config=configs/grpo_hypothesis.yaml \
    env=CrowRLEnv \
    dataset=nemotron_hypothesis \
    model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --num-gpus 8
```

### 6. BixBench MCQ 评估（原有功能）

```bash
MODEL=gpt-4o bash scripts/eval_bixbench.sh
```

---

## 组件来源

| 组件 | 来源 | 状态 |
|------|------|------|
| `aviary.envs.notebook.NBEnvironment` | FutureHouse/aviary | ✅ 真实 API |
| `ldp.agent.ReActAgent` | FutureHouse/aviary/ldp | ✅ 真实 API |
| `fhda.DataAnalysisEnv` | FutureHouse/data-analysis-crow | ✅ 真实 API |
| `nvidia/Nemotron-RL-bixbench_hypothesis` | HuggingFace | ✅ 真实数据集 |
| `verifiers.GRPOTrainer` | alexandonian/verifiers | ✅ 真实框架 |
| NeMo RL (`ng_run`) | NVIDIA-NeMo/RL | ⚠️ 需要 GPU |

---

## 参考

- Nemotron 数据集: https://huggingface.co/datasets/nvidia/Nemotron-RL-bixbench_hypothesis
- data-analysis-crow: https://github.com/Future-House/data-analysis-crow
- verifiers 框架: https://github.com/alexandonian/verifiers
- GRPO 训练思路: https://blog.ando.ai/posts/ai-grpo/
- NVIDIA 官方教程: https://developer.nvidia.com/blog/how-to-train-scientific-agents-with-reinforcement-learning/
- BixBench: https://github.com/Future-House/BixBench — arXiv:2503.00096
- Aviary: https://github.com/Future-House/aviary — arXiv:2412.21154
