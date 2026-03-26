"""
src/verifiers/rubric.py
========================
StepRubric — aggregates per-step verifier scores into:

  1. A trajectory-level reward (weighted sum of step scores).
  2. Per-step rewards for step-level GRPO advantage computation.

The rubric is compatible with the reward function signature expected by the
`verifiers` library (prompt, completion, answer, **kwargs) → float, as well as
the async batch-scoring API used in the training loop.

Usage example
-------------
    from src.verifiers import StepRubric, HYPOTHESIS_PROTOCOL

    rubric = StepRubric(protocol=HYPOTHESIS_PROTOCOL)

    # Score a complete trajectory
    step_scores = rubric.score_trajectory(trajectory)
    total_reward = rubric.aggregate(step_scores)

    # Use as a verifiers-style reward function (for vf.Rubric / vf.GRPOTrainer)
    reward = rubric.reward_func(prompt, completion, answer, state=state)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.verifiers.protocol import AnalysisProtocol, HYPOTHESIS_PROTOCOL
from src.verifiers.step_verifiers import (
    AnswerSubmissionVerifier,
    BaseStepVerifier,
    DataLoadingVerifier,
    ExploratoryAnalysisVerifier,
    HypothesisUnderstandingVerifier,
    InterpretationVerifier,
    StatisticalTestingVerifier,
)

logger = logging.getLogger(__name__)

# Canonical mapping: protocol step name → verifier class
_DEFAULT_VERIFIERS: dict[str, type[BaseStepVerifier]] = {
    "hypothesis_understanding": HypothesisUnderstandingVerifier,
    "data_loading": DataLoadingVerifier,
    "exploratory_analysis": ExploratoryAnalysisVerifier,
    "statistical_testing": StatisticalTestingVerifier,
    "interpretation": InterpretationVerifier,
    "answer_submission": AnswerSubmissionVerifier,
}


class StepRubric:
    """
    Evaluates an agent trajectory against an AnalysisProtocol.

    Each protocol step is scored by the matching BaseStepVerifier.  The
    trajectory-level reward is the weighted sum of per-step scores.

    Parameters
    ----------
    protocol :
        Protocol defining steps and their weights.
    verifier_map :
        Optional override mapping step names to verifier *instances*.
        Steps not present in the map fall back to _DEFAULT_VERIFIERS.
    """

    def __init__(
        self,
        protocol: AnalysisProtocol = HYPOTHESIS_PROTOCOL,
        verifier_map: dict[str, BaseStepVerifier] | None = None,
    ) -> None:
        self.protocol = protocol
        self._verifiers: dict[str, BaseStepVerifier] = {}

        for step in protocol:
            if verifier_map and step.name in verifier_map:
                self._verifiers[step.name] = verifier_map[step.name]
            elif step.name in _DEFAULT_VERIFIERS:
                self._verifiers[step.name] = _DEFAULT_VERIFIERS[step.name]()
            else:
                logger.warning("No verifier registered for protocol step '%s'", step.name)

    # ------------------------------------------------------------------
    # Core scoring methods
    # ------------------------------------------------------------------

    def score_step(
        self,
        step_name: str,
        step_content: str,
        state: dict[str, Any],
    ) -> float:
        """Score a single step and return a value in [0, 1]."""
        verifier = self._verifiers.get(step_name)
        if verifier is None:
            logger.debug("No verifier for step '%s', returning 0.0", step_name)
            return 0.0
        try:
            return float(verifier.score(step_content, state))
        except Exception:
            logger.exception("Verifier error for step '%s'", step_name)
            return 0.0

    def score_trajectory(
        self,
        trajectory: dict[str, Any],
    ) -> dict[str, float]:
        """
        Score every step in a trajectory dict.

        Expected trajectory format (produced by run_episode in notebook_agent.py
        and analogous functions in crow_env.py):

            {
                "steps": [
                    {
                        "step": 1,
                        "action": "<assistant message text>",
                        "tool_result": "<env response text>",
                        "reward": 0.0,   # may be overwritten by rubric
                    },
                    ...
                ],
                "hypothesis": "...",
                "ground_truth": "True" | "False",
                "submitted_answer": "True" | "False" | None,
            }

        Returns
        -------
        dict mapping protocol step name → score in [0, 1]
        """
        step_scores: dict[str, float] = {s.name: 0.0 for s in self.protocol}
        steps = trajectory.get("steps", [])
        if not steps:
            return step_scores

        # Build a rolling state so later verifiers can see earlier results
        state: dict[str, Any] = {
            "hypothesis": trajectory.get("hypothesis", ""),
            "ground_truth": trajectory.get("ground_truth", ""),
            "submitted_answer": trajectory.get("submitted_answer"),
            "tool_results": [],
        }

        # Concatenate all step content for "global" verifiers that look at the
        # full conversation (e.g. interpretation, answer submission)
        full_content = "\n\n".join(str(s.get("action", "")) for s in steps)

        # Accumulate tool results for use in step verifiers
        tool_results: list[str] = []
        for s in steps:
            if s.get("tool_result"):
                tool_results.append(str(s["tool_result"]))

        # Score each protocol step against the most relevant portion of the
        # trajectory.  We use a simple heuristic: match step to the part of the
        # conversation that most likely contains that step's content.
        step_texts = _map_conversation_to_protocol(steps, self.protocol)

        for proto_step in self.protocol:
            content = step_texts.get(proto_step.name, full_content)
            state["tool_results"] = tool_results  # verifiers can inspect this
            score = self.score_step(proto_step.name, content, state)
            step_scores[proto_step.name] = score

        return step_scores

    def aggregate(self, step_scores: dict[str, float]) -> float:
        """Compute the weighted trajectory-level reward from per-step scores."""
        total = 0.0
        for step in self.protocol:
            total += step.weight * step_scores.get(step.name, 0.0)
        return total

    def score_full_trajectory(self, trajectory: dict[str, Any]) -> float:
        """Convenience: score a trajectory and return the scalar reward."""
        return self.aggregate(self.score_trajectory(trajectory))

    # ------------------------------------------------------------------
    # verifiers-library compatible reward function
    # ------------------------------------------------------------------

    def reward_func(
        self,
        prompt: str | list[dict],
        completion: str | list[dict],
        answer: str | None = None,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Reward function compatible with `verifiers.Rubric`.

        Converts a (prompt, completion, answer) tuple into a trajectory dict
        and scores it with the full protocol rubric.
        """
        if state is None:
            state = {}

        # Extract text from message lists if needed
        comp_text = _messages_to_text(completion)

        trajectory: dict[str, Any] = {
            "steps": _text_to_steps(comp_text),
            "hypothesis": state.get("hypothesis", _extract_hypothesis(prompt)),
            "ground_truth": answer or state.get("ground_truth", ""),
            "submitted_answer": state.get("submitted_answer")
            or _extract_submitted_answer(comp_text),
        }
        return self.score_full_trajectory(trajectory)

    #: Reward per XML format tag found (format compliance bonus)
    FORMAT_TAG_BONUS: float = 0.1

    def format_reward_func(
        self,
        prompt: str | list[dict],
        completion: str | list[dict],
        answer: str | None = None,
        **kwargs: Any,
    ) -> float:
        """
        Bonus reward for following the XML/think format expected by the system.
        (+FORMAT_TAG_BONUS per required XML tag found)
        """
        comp_text = _messages_to_text(completion)
        score = 0.0
        for tag in ("<think>", "</think>", "<answer>", "</answer>"):
            if tag in comp_text:
                score += self.FORMAT_TAG_BONUS
        return min(1.0, score)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _messages_to_text(messages: str | list[dict]) -> str:
    """Flatten a message list to a single string."""
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def _text_to_steps(text: str) -> list[dict[str, Any]]:
    """Very lightweight heuristic: split text into 'steps' by assistant turn."""
    steps = []
    for i, chunk in enumerate(re.split(r"\[assistant\]:", text), start=1):
        chunk = chunk.strip()
        if chunk:
            steps.append({"step": i, "action": chunk, "tool_result": ""})
    return steps or [{"step": 1, "action": text, "tool_result": ""}]


def _extract_hypothesis(prompt: str | list[dict]) -> str:
    """Try to extract the hypothesis from the prompt."""
    text = _messages_to_text(prompt)
    m = re.search(r"(?i)hypothesis[:\s]+(.+?)(?:\n|$)", text)
    return m.group(1).strip() if m else ""


def _extract_submitted_answer(text: str) -> str | None:
    """Extract the last submit_answer call from text."""
    m = re.findall(r'submit_answer\s*\(\s*["\']?(true|false)["\']?\s*\)', text, re.IGNORECASE)
    return m[-1].capitalize() if m else None


def _map_conversation_to_protocol(
    steps: list[dict[str, Any]],
    protocol: AnalysisProtocol,
) -> dict[str, str]:
    """
    Heuristically assign conversation steps to protocol steps.

    Strategy: assign each protocol step to the conversation step whose
    content best matches that step's keywords.  If no conversation step
    matches, fall back to the full concatenated content.
    """
    proto_names = [s.name for s in protocol]
    full_text = "\n\n".join(str(s.get("action", "")) for s in steps)
    result: dict[str, str] = {name: full_text for name in proto_names}

    if not steps:
        return result

    # Score each (proto_step, conv_step) pair by keyword overlap
    for ps in protocol:
        if not ps.keywords:
            continue
        best_score = -1.0
        best_text = full_text
        for conv_step in steps:
            content = str(conv_step.get("action", ""))
            tool_result = str(conv_step.get("tool_result", ""))
            combined = content + " " + tool_result
            hit_count = sum(
                1 for kw in ps.keywords if kw.lower() in combined.lower()
            )
            if hit_count > best_score:
                best_score = hit_count
                best_text = combined
        if best_score > 0:
            result[ps.name] = best_text

    return result
