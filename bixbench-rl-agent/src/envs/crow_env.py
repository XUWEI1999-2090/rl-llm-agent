"""
src/envs/crow_env.py
=====================
data-analysis-crow environment wrapper for RL training on the
Nemotron-RL-bixbench_hypothesis dataset.

This module bridges two worlds:
  * FutureHouse data-analysis-crow (DataAnalysisEnv / fhda)  — async aviary env
  * RL training loop (train_grpo_hypothesis.py) — needs rollouts + step rewards

Key design decisions
--------------------
1.  CrowRLEnv wraps DataAnalysisEnv and exposes a simple async `run_episode()`
    that produces a trajectory dict compatible with StepRubric.

2.  The aviary tool system is used as-is; we do NOT try to bridge it with
    OpenAI-compatible function calling at this layer.  The verifiers library's
    GRPOTrainer can optionally be used for the training step (see
    train_grpo_hypothesis.py).

3.  Step-level GRPO advantage computation is done by StepLevelGRPOGrouper
    (from scripts/train_grpo.py) after collecting rollouts.

References
----------
* DataAnalysisEnv: https://github.com/Future-House/data-analysis-crow
* CapsuleDataset.get_new_env_by_idx: fhda/dataset.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for hypothesis testing (mirrors fhda.prompts)
# ---------------------------------------------------------------------------

HYPOTHESIS_SYSTEM_PROMPT = """\
You are a rigorous scientific data analyst.  Your task is to determine whether a
biological hypothesis is supported or refuted by the available data.

Follow this protocol strictly:
1. Load and inspect the data files available in the working directory.
2. Perform exploratory data analysis to understand the structure and quality.
3. Apply the most appropriate statistical test to evaluate the hypothesis.
4. Interpret the p-value, effect size, and direction of the effect.
5. Call submit_answer("True") if the hypothesis is supported, or
   submit_answer("False") if it is refuted.

Always show your reasoning step by step before drawing a conclusion.
"""


# ---------------------------------------------------------------------------
# CrowRLEnv
# ---------------------------------------------------------------------------


class CrowRLEnv:
    """
    Async wrapper around DataAnalysisEnv (from fhda) for RL training.

    Parameters
    ----------
    problem_id    : Unique identifier for the problem (e.g. capsule_id).
    hypothesis    : The hypothesis string.
    ground_truth  : "True" or "False" — the correct answer.
    work_dir      : Directory containing the capsule data files.
    use_docker    : Whether to run the notebook kernel inside Docker.
    docker_image  : Docker image tag for the BixBench execution environment.
    correct_reward: Scalar reward for a correct final answer (default 1.0).
    """

    def __init__(
        self,
        *,
        problem_id: str,
        hypothesis: str,
        ground_truth: str,
        work_dir: str | Path,
        use_docker: bool = True,
        docker_image: str = "futurehouse/bixbench:aviary-notebook-env",
        correct_reward: float = 1.0,
    ) -> None:
        self.problem_id = problem_id
        self.hypothesis = hypothesis
        self.ground_truth = ground_truth
        self.work_dir = Path(work_dir)
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.correct_reward = correct_reward
        self._env = None

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    async def _make_env(self):
        """Instantiate DataAnalysisEnv from data-analysis-crow."""
        try:
            from fhda.data_analysis_env import DataAnalysisEnv
            from fhda.utils import NBLanguage
            from aviary.core import EvalAnswerMode
        except ImportError as exc:
            raise ImportError(
                "data-analysis-crow (fhda) is not installed. "
                "Run: pip install -e data-analysis-crow/ "
                "or see scripts/setup_env.sh."
            ) from exc

        nb_path = self.work_dir / "notebook.ipynb"
        problem = self._format_problem()

        env = DataAnalysisEnv(
            problem_id=self.problem_id,
            problem=problem,
            answer=self.ground_truth,
            eval_mode=EvalAnswerMode.EXACT,
            nb_path=nb_path,
            work_dir=self.work_dir,
            language=NBLanguage.PYTHON,
            system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
            correct_reward=self.correct_reward,
        )
        return env

    def _format_problem(self) -> str:
        return (
            f"Evaluate the following biological hypothesis using the data in "
            f"the working directory.\n\n"
            f"**Hypothesis:** {self.hypothesis}\n\n"
            f"Analyse the data, apply the appropriate statistical test, and "
            f"call `submit_answer(\"True\")` if the hypothesis is supported "
            f"by the data or `submit_answer(\"False\")` if it is refuted."
        )

    async def reset(self):
        """Reset the environment and return (messages, tools)."""
        self._env = await self._make_env()
        return await self._env.reset()

    async def step(self, action):
        """Execute one action and return (messages, reward, done, truncated)."""
        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        return await self._env.step(action)

    async def close(self):
        if self._env is not None:
            await self._env.close()
            self._env = None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_crow_episode(
    agent,
    env: CrowRLEnv,
    max_steps: int = 30,
) -> dict[str, Any]:
    """
    Run a single hypothesis-testing episode.

    Returns a trajectory dict compatible with StepRubric.score_trajectory():

        {
            "capsule_id": str,
            "hypothesis": str,
            "ground_truth": str,
            "submitted_answer": str | None,
            "steps": [
                {
                    "step": int,
                    "action": str,          # assistant message text
                    "tool_result": str,     # env response text
                    "reward": float,        # per-step reward (0 until final step)
                    "log_prob": float|None,
                    "done": bool,
                }
            ],
            "total_reward": float,
            "n_steps": int,
            "done": bool,
        }
    """
    # Import agent utilities (ldp-based ReActAgent)
    from ldp.agent import ReActAgent
    from aviary.core import ToolRequestMessage

    obs, tools = await env.reset()

    # Initialise agent state
    if hasattr(agent, "init_state"):
        agent_state = await agent.init_state(tools)
    else:
        agent_state = None

    trajectory: dict[str, Any] = {
        "capsule_id": env.problem_id,
        "hypothesis": env.hypothesis,
        "ground_truth": env.ground_truth,
        "submitted_answer": None,
        "steps": [],
        "total_reward": 0.0,
        "n_steps": 0,
        "done": False,
    }

    for step_i in range(1, max_steps + 1):
        # Agent decision
        action_result, agent_state, _ = await agent.get_asv(agent_state, obs)
        action = action_result.value

        # Environment step
        obs, reward, done, truncated = await env.step(action)

        # Extract tool result from observations
        tool_result_text = _extract_tool_result(obs)
        submitted = _extract_submitted_answer_from_obs(obs)
        if submitted is not None:
            trajectory["submitted_answer"] = submitted

        step_data = {
            "step": step_i,
            "action": str(action),
            "tool_result": tool_result_text,
            "reward": reward,
            "log_prob": _extract_logprob(action_result),
            "done": done,
        }
        trajectory["steps"].append(step_data)
        trajectory["total_reward"] += reward
        trajectory["n_steps"] = step_i

        if done or truncated:
            trajectory["done"] = done
            break

    await env.close()
    return trajectory


# ---------------------------------------------------------------------------
# CrowDataset — thin wrapper over NemotronHypothesisDataset → CrowRLEnv
# ---------------------------------------------------------------------------


class CrowDataset:
    """
    Maps Nemotron dataset samples to CrowRLEnv instances.

    Parameters
    ----------
    nemotron_dataset : NemotronHypothesisDataset
        Loaded Nemotron hypothesis dataset.
    use_docker       : Whether to use Docker for notebook execution.
    docker_image     : Docker image for the execution environment.
    """

    def __init__(
        self,
        nemotron_dataset,
        use_docker: bool = True,
        docker_image: str = "futurehouse/bixbench:aviary-notebook-env",
    ) -> None:
        self.dataset = nemotron_dataset
        self.use_docker = use_docker
        self.docker_image = docker_image

    def get_env(self, idx: int) -> CrowRLEnv:
        sample = self.dataset[idx]
        work_dir = self.dataset.prepare_work_dir(sample)
        return CrowRLEnv(
            problem_id=sample.capsule_id,
            hypothesis=sample.hypothesis,
            ground_truth=sample.answer,
            work_dir=work_dir,
            use_docker=self.use_docker,
            docker_image=self.docker_image,
        )

    def __len__(self) -> int:
        return len(self.dataset)


# ---------------------------------------------------------------------------
# Batch rollout collector
# ---------------------------------------------------------------------------


async def collect_crow_rollouts(
    crow_dataset: CrowDataset,
    agent,
    rubric,
    n_samples: int = 4,
    n_parallel: int = 8,
    max_steps: int = 30,
) -> list[dict[str, Any]]:
    """
    Collect rollouts from the CrowDataset, score them with the StepRubric,
    and return trajectory dicts with `step_scores` and `total_reward` filled in.
    """
    import random

    indices = random.sample(range(len(crow_dataset)), min(n_samples, len(crow_dataset)))

    tasks = [
        run_crow_episode(agent, crow_dataset.get_env(idx), max_steps=max_steps)
        for idx in indices
        for _ in range(n_parallel)
    ]

    logger.info(
        "Collecting %d rollouts (%d samples × %d parallel)...",
        len(tasks),
        len(indices),
        n_parallel,
    )

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
            # Score with rubric and store per-step scores
            step_scores = rubric.score_trajectory(r)
            r["step_scores"] = step_scores
            r["total_reward"] = rubric.aggregate(step_scores)
            trajectories.append(r)

    if trajectories:
        mean_reward = sum(t["total_reward"] for t in trajectories) / len(trajectories)
        logger.info(
            "Collected %d/%d rollouts, mean_reward=%.3f",
            len(trajectories),
            len(tasks),
            mean_reward,
        )

    return trajectories


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_tool_result(obs) -> str:
    """Extract text from an aviary observations list."""
    if not obs:
        return ""
    try:
        from aviary.core import Message

        texts = []
        for msg in obs:
            if hasattr(msg, "content") and msg.content:
                texts.append(str(msg.content))
        return "\n".join(texts)
    except Exception:
        return str(obs)


def _extract_submitted_answer_from_obs(obs) -> str | None:
    """Check if the last observation contains a submit_answer confirmation."""
    text = _extract_tool_result(obs)
    import re

    m = re.search(r"submitted answer:\s*(true|false)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def _extract_logprob(action_result) -> float | None:
    """Extract log probability from ldp OpResult."""
    try:
        if hasattr(action_result, "log_prob"):
            return action_result.log_prob
        if hasattr(action_result, "extras") and action_result.extras:
            return action_result.extras.get("log_prob")
    except Exception:
        pass
    return None
