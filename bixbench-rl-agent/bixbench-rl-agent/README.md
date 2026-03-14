# BixBench RL Agent

Edison Scientific / NeMo Gym + Aviary + BixBench 工作的完整复现项目。

---

## ⚠️ 关于本项目的诚实说明

本项目基于**真实读取的源码**构建，而非猜测：

| 组件 | 来源 | 状态 |
|------|------|------|
| `aviary.envs.notebook.NBEnvironment` | 读取了 `/usr/local/lib/python3.12/dist-packages/aviary/envs/notebook/env.py` | ✅ 真实 API |
| `ldp.agent.ReActAgent` | 读取了 `ldp/agent/react_agent.py` | ✅ 真实 API |
| `ldp.agent.SimpleAgentState.hide_old_env_states` | 读取了源码确认此参数存在 | ✅ 真实参数 |
| NeMo Gym / NeMo RL | 未安装，基于官方文档 | ⚠️ 接口层面 |
| BixBench 下载 / 评估 | 未安装，基于论文和文档 | ⚠️ 接口层面 |

**尚未端到端运行**：需要真实 GPU 机器 + 完整安装（见下方步骤）。

---

## 架构

```
ControlledNotebookEnv (src/envs/notebook_env.py)
  ↑ 继承自
aviary.envs.notebook.NBEnvironment          ← 真实源码已读取并确认
  工具: edit_cell(contents, idx)             ← 真实签名
         list_workdir()                       ← 真实签名
  + 新增: submit_answer(answers)             ← BixBench 专用

ldp.ReActAgent(                              ← 真实源码已读取并确认
    hide_old_env_states=True,               ← Edison 创新 #1 的真实实现
    hide_old_action_content=True,
    sliding_window=None,
)

NeMo RL GRPO (configs/grpo_step.yaml)
    group_size: 1                            ← Edison 创新 #2
    use_transitions: true
    reward_type: step_level
```

## Edison 两个创新的真实实现位置

### 创新 #1：上下文截断

**在 ldp 源码中已内置**，不需要自己实现：

```python
# ldp/agent/simple_agent.py — SimpleAgentState.get_next_state()
class HiddenEnvStateMessage(EnvStateMessage):
    content: str = "[Previous environment state - hidden]"
```

只需配置：
```python
ReActAgent(
    hide_old_env_states=True,    # 自动替换旧 Notebook 状态为 "[hidden]"
    hide_old_action_content=True,
)
```

### 创新 #2：步骤级 GRPO

在 NeMo RL 的 YAML 配置中：
```yaml
grpo:
  group_size: 1          # 单步分组，而非整条轨迹
  use_transitions: true  # 支持状态转换训练
  reward_type: step_level
```

---

## 快速开始

### 1. 搭建环境

```bash
# 克隆本项目
git clone <this-repo>
cd bixbench-rl-agent

# 一键安装所有真实组件
bash scripts/setup_env.sh
```

`setup_env.sh` 会克隆并安装：
- `github.com/Future-House/aviary` — 真实 NBEnvironment
- `github.com/Future-House/BixBench` — 数据集 + 评估 harness
- `github.com/NVIDIA-NeMo/Gym` — rollout 扩展
- `github.com/NVIDIA-NeMo/RL` — GRPO 训练
- `docker pull futurehouse/bixbench-env:v1.0` — 生物信息学执行环境

### 2. 评估（不需要 GPU）

```bash
export OPENAI_API_KEY=sk-...
bash scripts/eval_bixbench.sh
```

### 3. RL 训练（需要 8×H100）

```bash
# 启动 vLLM 服务
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --port 10240 --max-model-len 262144 --enable-auto-tool-choice

# 收集 rollout（测试）
ng_collect_rollouts \
    +agent_name=notebook_aviary_agent \
    +input_jsonl_fpath=data/bixbench/examples.jsonl

# 启动 GRPO 训练
ng_run \
    +config_paths=[configs/notebook_aviary.yaml,configs/model.yaml] \
    trainer.grpo_config=configs/grpo_step.yaml \
    env=ControlledNotebookEnv \
    dataset=bixbench_train \
    model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --num-gpus 8
```

---

## 参考

- NVIDIA 官方教程: https://developer.nvidia.com/blog/how-to-train-scientific-agents-with-reinforcement-learning/
- NeMo Gym: https://github.com/NVIDIA-NeMo/Gym
- NeMo RL: https://github.com/NVIDIA-NeMo/RL
- Aviary: https://github.com/Future-House/aviary — arXiv:2412.21154
- BixBench: https://github.com/Future-House/BixBench — arXiv:2503.00096
