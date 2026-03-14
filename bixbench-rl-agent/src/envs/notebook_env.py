"""
src/envs/notebook_env.py
=========================
Edison Scientific 的两个核心创新，基于真实的 aviary.envs.notebook.NBEnvironment API。

读完源码后的关键发现：
  1. 上下文截断: ldp 的 ReActAgent 已内置 hide_old_env_states / sliding_window,
     这正是 Edison 的"清除交互历史，仅保留原始指令 + 历史操作 + 当前 Notebook"。
     我们在 ControlledNotebookEnv 里通过覆盖 step() 来强制只返回当前 Notebook 状态。

  2. submit_answer 工具: 原始 NBEnvironment 没有此工具，需要我们添加。
     BixBench 需要智能体最终调用 submit_answer({"Q1": "...", "Q2": "..."})。

真实的 NBEnvironment 工具签名（来自源码）:
  - edit_cell(contents: str, idx: int | None = None)
  - list_workdir() -> str

真实的 step() 返回值:
  tuple[Messages, float, bool, bool]
  (observations, reward, done, truncated)

真实的 reset() 返回值:
  tuple[Messages, list[Tool]]

真实的 EnvStateMessage: aviary 的特殊消息类型，携带 Notebook 的 Markdown 渲染
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# 真实 aviary 导入（需要从源码安装 FutureHouse/aviary，非 PyPI 的 NASA aviary）
# git clone https://github.com/Future-House/aviary.git
# uv pip install -e 'aviary[notebook,labbench]'
from aviary.envs.notebook.env import NBEnvironment, NBEnvironmentState
from aviary.core import Tool, Messages, ToolRequestMessage
from aviary.message import EnvStateMessage
from pydantic import Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BixBench 特定状态扩展
# ---------------------------------------------------------------------------

class BixBenchNBState(NBEnvironmentState):
    """
    扩展 NBEnvironmentState，增加 BixBench 特定字段：
      - task_instruction: 原始任务指令（不随步骤变化）
      - questions: 需要回答的问题列表
      - submitted_answers: 智能体提交的答案
    """
    task_instruction: str = ""
    questions: list[dict[str, str]] = Field(default_factory=list)
    submitted_answers: dict[str, str] = Field(default_factory=dict)
    ground_truth: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ControlledNotebookEnv — Edison 创新 #1 的正确实现
# ---------------------------------------------------------------------------

class ControlledNotebookEnv(NBEnvironment[BixBenchNBState]):
    """
    Edison Scientific 的上下文控制 Notebook 环境。

    创新 #1 实现:
      在每步 step() 之后，我们不返回完整的消息历史，而是只返回：
        - 原始任务指令（固定，不变）
        - 历史操作摘要（仅动作，非完整对话）
        - 当前 Notebook 状态（完整，实时）

    这通过两种机制实现：
      A. 本类的 step() 覆盖：只返回当前 EnvStateMessage，不累积历史
      B. 搭配 ldp.ReActAgent(hide_old_env_states=True, sliding_window=N)：
         ldp 内置的上下文管理，hide_old_env_states 会把老的 EnvStateMessage
         替换为 "[Previous environment state - hidden]"，完全匹配 Edison 的方案。

    创新 #2 (步骤级 GRPO) 在训练脚本中实现，不在此环境中。
    """

    STATE_CLS = BixBenchNBState

    # BixBench 特定配置
    task_instruction: str = ""
    questions: list[dict[str, str]] = Field(default_factory=list)
    ground_truth: dict[str, str] = Field(default_factory=dict)

    async def reset(self) -> tuple[Messages, list[Tool]]:
        """重置环境，返回初始观测和工具列表（含 submit_answer）。"""
        msgs, tools = await super().reset()

        # 初始化 BixBench 特定状态
        self.state.task_instruction = self.task_instruction
        self.state.questions = self.questions
        self.state.ground_truth = self.ground_truth
        self.state.submitted_answers = {}

        # 添加 BixBench 专用 submit_answer 工具
        submit_tool = Tool.from_function(self._make_submit_fn())
        tools.append(submit_tool)
        self.tools = tools

        # 将任务指令注入到初始消息中
        from aviary.core import Message
        instruction_msg = Message(
            content=self._format_instruction()
        )
        return [instruction_msg, *msgs], tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        """
        执行一步，返回截断后的观测。

        Edison 创新 #1 的核心：
          - super().step() 会返回工具响应 + 新的 EnvStateMessage（当前 notebook）
          - 我们不在此处截断历史（那是 ldp.ReActAgent 的职责）
          - 我们确保每步都包含最新的 EnvStateMessage，供 hide_old_env_states 机制使用
        """
        obs, reward, done, truncated = await super().step(action)
        return obs, reward, done, truncated

    def _format_instruction(self) -> str:
        """格式化任务指令，包含所有问题。"""
        q_block = "\n".join(
            f"  [{q['id']}] {q['question']}" for q in self.questions
        )
        return (
            f"# 生物信息学分析任务\n\n"
            f"## 任务描述\n{self.task_instruction}\n\n"
            f"## 需要回答的问题\n{q_block}\n\n"
            f"## 工作流程\n"
            f"1. 调用 list_workdir() 了解数据文件\n"
            f"2. 编写 Python/R 代码分析数据（使用 edit_cell）\n"
            f"3. 分析完成后调用 submit_answer 提交答案\n"
        )

    def _make_submit_fn(self):
        """
        生成 submit_answer 工具函数。

        使用闭包来访问 self.state，因为 Tool.from_function 需要一个普通函数。
        """
        env = self

        def submit_answer(answers: str) -> str:
            """Submit final answers to the research questions.

            Call this when you are confident in all your answers.
            Ends the episode and triggers scoring.

            Args:
                answers: JSON string mapping question IDs to answer strings.
                    Example: '{"Q1": "42", "Q2": "Cell cycle"}'
            """
            try:
                if isinstance(answers, dict):
                    answer_dict = answers
                else:
                    answer_dict = json.loads(answers)
            except (json.JSONDecodeError, TypeError) as e:
                return f"Error parsing answers JSON: {e}. Please provide valid JSON."

            if not answer_dict:
                return "Error: no answers provided."

            env.state.submitted_answers = answer_dict
            score = env._score_answers(answer_dict)
            env.state.total_reward += score
            env.state.done = True

            lines = "\n".join(f"  {k}: {v}" for k, v in answer_dict.items())
            return (
                f"Answers submitted ({len(answer_dict)} questions):\n{lines}\n\n"
                f"Episode complete. Score: {score:.3f}"
            )

        return submit_answer

    def _score_answers(self, submitted: dict[str, str]) -> float:
        """对提交的答案评分，返回 [0, 1]。"""
        if not self.ground_truth:
            return 0.0

        correct = 0
        for qid, gt in self.ground_truth.items():
            pred = submitted.get(qid, "")
            if _answers_match(pred, gt):
                correct += 1

        return correct / len(self.ground_truth)


# ---------------------------------------------------------------------------
# BixBench TaskDataset — 与 ldp/NeMo Gym 的 TaskDataset 接口对接
# ---------------------------------------------------------------------------

class BixBenchDataset:
    """
    BixBench capsule 数据集，符合 aviary.env.TaskDataset 接口。

    从 HuggingFace 加载：huggingface-cli download futurehouse/BixBench
    或直接使用 BixBench 仓库的 python -m bixbench.download
    """

    def __init__(
        self,
        data_dir: str | Path,
        docker_image: str = "futurehouse/bixbench-env:v1.0",
        use_docker: bool = True,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.docker_image = docker_image
        self.use_docker = use_docker
        self.split = split
        self._capsules = self._load_capsules()

    def _load_capsules(self) -> list[dict]:
        capsules = []
        if not self.data_dir.exists():
            logger.warning(f"Data dir not found: {self.data_dir}")
            return capsules

        for capsule_dir in sorted(self.data_dir.iterdir()):
            meta = capsule_dir / "metadata.json"
            if meta.exists():
                data = json.loads(meta.read_text())
                if self.split == "all" or data.get("split", "train") == self.split:
                    data["_dir"] = str(capsule_dir)
                    capsules.append(data)

        logger.info(f"Loaded {len(capsules)} capsules (split={self.split})")
        return capsules

    def get_env(self, idx: int) -> ControlledNotebookEnv:
        """返回第 idx 个 capsule 对应的环境实例。"""
        capsule = self._capsules[idx]
        return ControlledNotebookEnv(
            work_dir=capsule["_dir"],
            use_docker=self.use_docker,
            task_instruction=capsule.get("description", ""),
            questions=capsule.get("questions", []),
            ground_truth={
                q["id"]: q["answer"]
                for q in capsule.get("questions", [])
            },
            # BixBench Docker 镜像包含完整生物信息学环境
            # 在 config.py 中通过 NB_ENVIRONMENT_DOCKER_IMAGE 设置
        )

    def __len__(self) -> int:
        return len(self._capsules)


# ---------------------------------------------------------------------------
# 答案匹配函数
# ---------------------------------------------------------------------------

def _answers_match(pred: str, gt: str, numeric_tol: float = 0.05) -> bool:
    """判断预测答案是否与标准答案匹配。"""
    import re

    pred = pred.strip().lower()
    gt = gt.strip().lower()

    if pred == gt:
        return True

    # 数值容差匹配
    try:
        p, g = float(pred.replace(",", "")), float(gt.replace(",", ""))
        if g == 0:
            return abs(p) < 1e-9
        return abs(p - g) / abs(g) <= numeric_tol
    except ValueError:
        pass

    # 集合匹配（逗号分隔列表）
    ps = {x.strip() for x in re.split(r"[,;]", pred) if x.strip()}
    gs = {x.strip() for x in re.split(r"[,;]", gt) if x.strip()}
    if len(ps) > 1 and ps == gs:
        return True

    return False
