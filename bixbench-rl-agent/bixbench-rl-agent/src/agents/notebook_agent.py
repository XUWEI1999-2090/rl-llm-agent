"""
src/agents/notebook_agent.py
=============================
基于真实 ldp.ReActAgent API 的 BixBench Notebook 智能体。

关键发现（读完 ldp 源码后）：
  - ldp.ReActAgent 有内置的 hide_old_env_states 参数
  - hide_old_env_states=True 会把老的 EnvStateMessage 替换为 "[Previous env state - hidden]"
  - 这正是 Edison 上下文截断的实现机制，无需自己实现！
  - sliding_window=N 控制保留多少步的历史（None = 全部）

Edison 创新 #1 的正确配置：
  ReActAgent(
      hide_old_env_states=True,   ← 丢弃旧的 Notebook 状态（只保留最新）
      hide_old_action_content=True, ← 压缩旧的动作消息
      sliding_window=None,          ← 保留所有动作（但 env state 被隐藏）
  )

这样在每步，智能体看到的是：
  [system prompt]
  [原始指令]
  [Step 1 动作] + [HIDDEN env state]
  [Step 2 动作] + [HIDDEN env state]
  ...
  [Step N-1 动作] + [HIDDEN env state]
  [Step N-1 env state — VISIBLE, 最新的 notebook]
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

# 真实的 ldp 导入
# pip install ldp（来自 FutureHouse，已随 fhaviary 安装）
from ldp.agent import ReActAgent
from ldp.agent.simple_agent import SimpleAgentState
from aviary.core import Message

from src.envs.notebook_env import ControlledNotebookEnv, BixBenchDataset

logger = logging.getLogger(__name__)


def make_bixbench_agent(
    model_name: str = "gpt-4o",
    model_base_url: str | None = None,
    temperature: float = 0.7,
    max_steps: int = 30,
) -> ReActAgent:
    """
    创建 BixBench 专用 ReAct 智能体。

    Edison 创新 #1 通过以下参数实现上下文控制：
      - hide_old_env_states=True: 只保留最新的 Notebook 状态
      - hide_old_action_content=True: 压缩旧动作的 content
      - sliding_window=None: 保留所有步骤（但 env state 被隐藏了）

    Args:
        model_name: LLM 模型名，例如 "gpt-4o" 或 vLLM 服务的模型名
        model_base_url: 本地 vLLM 服务的 base URL（None = 用 OpenAI）
        temperature: 采样温度（RL 训练时用 0.7，评估时用 0.3）
        max_steps: 最大步数

    Returns:
        配置好的 ReActAgent 实例
    """
    llm_config: dict[str, Any] = {
        "name": model_name,
        "temperature": temperature,
        "logprobs": True,      # GRPO 训练需要 log probs
        "top_logprobs": 1,
        "timeout": 120.0,
    }

    # 本地 vLLM 服务支持
    if model_base_url:
        llm_config["base_url"] = model_base_url

    return ReActAgent(
        llm_model=llm_config,
        # ── Edison 创新 #1: 上下文截断 ──────────────────────────────
        hide_old_env_states=True,       # 丢弃旧的 Notebook 状态
        hide_old_action_content=True,   # 压缩旧的动作内容
        sliding_window=None,            # 保留所有步骤记录（env state 已隐藏）
        # ────────────────────────────────────────────────────────────
        single_prompt=False,            # 双提示 ReAct（更稳定）
    )


async def run_episode(
    agent: ReActAgent,
    env: ControlledNotebookEnv,
    max_steps: int = 30,
) -> dict[str, Any]:
    """
    运行一个完整的 BixBench episode。

    Returns:
        包含轨迹信息的字典，用于 GRPO 训练数据收集
    """
    # Reset
    obs, tools = await env.reset()
    agent_state = await agent.init_state(tools)

    trajectory = {
        "capsule_id": getattr(env, "_capsule_id", "unknown"),
        "steps": [],
        "total_reward": 0.0,
        "done": False,
        "n_steps": 0,
    }

    for step_i in range(1, max_steps + 1):
        # 智能体决策
        action_result, agent_state, _ = await agent.get_asv(agent_state, obs)
        action = action_result.value

        # 环境执行
        obs, reward, done, truncated = await env.step(action)

        # 记录 step 数据（用于步骤级 GRPO）
        step_data = {
            "step": step_i,
            "action": str(action),
            "reward": reward,
            "done": done,
            # log_prob 从 action_result 中提取（ldp 的 OpResult）
            "log_prob": _extract_logprob(action_result),
        }
        trajectory["steps"].append(step_data)
        trajectory["total_reward"] += reward
        trajectory["n_steps"] = step_i

        if done or truncated:
            trajectory["done"] = done
            break

    await env.close()
    return trajectory


def _extract_logprob(action_result) -> float | None:
    """从 ldp OpResult 中提取 log probability（用于 GRPO）。"""
    try:
        # ldp 的 OpResult 存储 log_prob 在 .extras 或通过 compute_graph 计算
        if hasattr(action_result, "log_prob"):
            return action_result.log_prob
        # 从 LLM response 中提取
        if hasattr(action_result, "extras") and action_result.extras:
            return action_result.extras.get("log_prob")
    except Exception:
        pass
    return None


async def collect_rollouts(
    dataset: BixBenchDataset,
    agent: ReActAgent,
    n_capsules: int = 4,
    n_parallel: int = 8,
    max_steps: int = 30,
) -> list[dict]:
    """
    并行收集 rollout，用于 GRPO 训练。

    对每个 capsule 运行 n_parallel 次，生成多条轨迹用于步骤级分组。
    """
    import random

    # 采样 capsule
    indices = random.sample(range(len(dataset)), min(n_capsules, len(dataset)))

    # 并行任务：每个 capsule × n_parallel 次
    tasks = [
        run_episode(agent, dataset.get_env(idx), max_steps=max_steps)
        for idx in indices
        for _ in range(n_parallel)
    ]

    logger.info(
        "Collecting %d rollouts (%d capsules × %d parallel)...",
        len(tasks), len(indices), n_parallel,
    )

    # 并发执行（限制并发数避免资源耗尽）
    semaphore = asyncio.Semaphore(n_parallel)

    async def run_with_sem(coro):
        async with semaphore:
            return await coro

    results = await asyncio.gather(
        *[run_with_sem(t) for t in tasks],
        return_exceptions=True,
    )

    trajectories = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Rollout failed: %s", r)
        else:
            trajectories.append(r)

    logger.info(
        "Collected %d/%d rollouts, mean_reward=%.3f",
        len(trajectories),
        len(tasks),
        sum(t["total_reward"] for t in trajectories) / max(len(trajectories), 1),
    )
    return trajectories
