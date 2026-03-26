"""
src/verifiers/step_verifiers.py
================================
Per-step verifiers for the AnalysisProtocol.

Each verifier implements `score(step_content, state)` and returns a float in
[0, 1] indicating how well the agent performed the corresponding protocol step.

Design principles
-----------------
* Pure functions over message text — no external calls, no I/O.
* Conservative scoring: prefer under-counting to over-counting.
* Deterministic (no LLM judge calls at this layer; those belong in StepRubric).

Verifier catalogue
------------------
  HypothesisUnderstandingVerifier — Step 1
  DataLoadingVerifier             — Step 2
  ExploratoryAnalysisVerifier     — Step 3
  StatisticalTestingVerifier      — Step 4
  InterpretationVerifier          — Step 5
  AnswerSubmissionVerifier        — Step 6
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseStepVerifier(ABC):
    """Abstract base for all step verifiers."""

    #: Name that matches a ProtocolStep.name
    step_name: str = ""

    def __init__(self, partial_credit: bool = True) -> None:
        self.partial_credit = partial_credit

    @abstractmethod
    def score(self, step_content: str, state: dict[str, Any]) -> float:
        """Return a score in [0, 1] for the given step content.

        Args:
            step_content: The full text of the assistant turn at this step.
            state: Mutable trajectory state dict (may contain 'answer',
                   'tool_results', 'step', 'ground_truth', …).
        """

    def _keyword_score(self, text: str, keywords: tuple[str, ...]) -> float:
        """Fraction of keywords found (case-insensitive) in text."""
        if not keywords:
            return 0.0
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        return hits / len(keywords)

    def _has_code_block(self, text: str) -> bool:
        return bool(re.search(r"```[\s\S]+?```", text) or "edit_cell" in text)

    def _extract_code_blocks(self, text: str) -> list[str]:
        return re.findall(r"```(?:\w+)?\n?([\s\S]+?)```", text)


# ---------------------------------------------------------------------------
# Step 1 — Hypothesis understanding
# ---------------------------------------------------------------------------


class HypothesisUnderstandingVerifier(BaseStepVerifier):
    """
    Score: did the agent correctly identify and restate the hypothesis?

    Signals:
      +0.5  mentions the hypothesis text (keyword overlap ≥ 0.3)
      +0.3  identifies measurable variables or biological entities
      +0.2  states what test / comparison is needed
    """

    step_name = "hypothesis_understanding"

    # Patterns indicating the agent is parsing the hypothesis
    _VARIABLE_PATTERNS = re.compile(
        r"\b(gene|protein|expression|level|count|abundance|mutation|"
        r"variant|pathway|cell|sample|group|condition|treatment|control|"
        r"correlation|association|difference|ratio|fold.change|fold-change)\b",
        re.IGNORECASE,
    )
    _TEST_DIRECTION_PATTERNS = re.compile(
        r"\b(test|compare|evaluate|determine|assess|analyse|analyze|"
        r"investigate|examine|check|verify|validate)\b",
        re.IGNORECASE,
    )

    def score(self, step_content: str, state: dict[str, Any]) -> float:
        hypothesis = state.get("hypothesis", "")
        if not step_content:
            return 0.0

        score = 0.0

        # +0.5 if the agent references key words from the hypothesis
        if hypothesis:
            hyp_words = {w.lower() for w in re.findall(r"\w+", hypothesis) if len(w) > 4}
            content_words = {w.lower() for w in re.findall(r"\w+", step_content)}
            if hyp_words:
                overlap = len(hyp_words & content_words) / len(hyp_words)
                score += 0.5 * min(1.0, overlap / 0.3)

        # +0.3 if biological variables are mentioned
        var_hits = len(self._VARIABLE_PATTERNS.findall(step_content))
        score += 0.3 * min(1.0, var_hits / 3)

        # +0.2 if the agent signals intent to test
        if self._TEST_DIRECTION_PATTERNS.search(step_content):
            score += 0.2

        return min(1.0, score)


# ---------------------------------------------------------------------------
# Step 2 — Data loading
# ---------------------------------------------------------------------------


class DataLoadingVerifier(BaseStepVerifier):
    """
    Score: did the agent load data files from the work directory?

    Signals:
      +0.6  code block contains a recognised data-loading call
      +0.2  code executed (tool_result exists and no error keyword)
      +0.2  agent calls list_workdir to discover available files
    """

    step_name = "data_loading"

    _LOAD_PATTERNS = re.compile(
        r"\b(pd\.read_csv|pd\.read_excel|pd\.read_table|pd\.read_parquet|"
        r"pd\.read_json|np\.load|np\.loadtxt|open\(|load_dataset|"
        r"read_csv|read_excel)\b",
        re.IGNORECASE,
    )
    _DIR_LIST_PATTERNS = re.compile(
        r"\b(list_workdir|os\.listdir|os\.scandir|glob\.glob)\b", re.IGNORECASE
    )
    _ERROR_PATTERNS = re.compile(
        r"\b(Error|Traceback|exception|failed|not found|no such file)\b",
        re.IGNORECASE,
    )

    def score(self, step_content: str, state: dict[str, Any]) -> float:
        if not step_content:
            return 0.0

        score = 0.0
        code_blocks = self._extract_code_blocks(step_content)
        code_text = "\n".join(code_blocks) or step_content

        # +0.6 if a data-loading function is present in code
        if self._LOAD_PATTERNS.search(code_text):
            score += 0.6

        # +0.2 if tool result exists and no error was reported
        tool_results = state.get("tool_results", [])
        if tool_results:
            last_result = str(tool_results[-1])
            if not self._ERROR_PATTERNS.search(last_result):
                score += 0.2

        # +0.2 if agent inspects the working directory
        if self._DIR_LIST_PATTERNS.search(step_content):
            score += 0.2

        return min(1.0, score)


# ---------------------------------------------------------------------------
# Step 3 — Exploratory analysis
# ---------------------------------------------------------------------------


class ExploratoryAnalysisVerifier(BaseStepVerifier):
    """
    Score: did the agent explore the data structure and quality?

    Partial credit per distinct EDA technique detected.
    """

    step_name = "exploratory_analysis"

    _EDA_PATTERNS: list[tuple[re.Pattern, float]] = [
        (re.compile(r"\.(head|tail)\s*\(", re.I), 0.15),
        (re.compile(r"\.(describe|info)\s*\(", re.I), 0.20),
        (re.compile(r"\.shape\b", re.I), 0.10),
        (re.compile(r"\.(dtypes|dtype)\b", re.I), 0.10),
        (re.compile(r"\.(isnull|isna|notna)\s*\(\)", re.I), 0.15),
        (re.compile(r"\.(value_counts|nunique)\s*\(", re.I), 0.15),
        (re.compile(r"\b(hist|boxplot|violinplot|scatter|heatmap|pairplot)\b", re.I), 0.15),
    ]

    def score(self, step_content: str, state: dict[str, Any]) -> float:
        if not step_content:
            return 0.0

        code_blocks = self._extract_code_blocks(step_content)
        code_text = "\n".join(code_blocks) or step_content

        score = 0.0
        for pattern, weight in self._EDA_PATTERNS:
            if pattern.search(code_text):
                score += weight

        return min(1.0, score)


# ---------------------------------------------------------------------------
# Step 4 — Statistical testing
# ---------------------------------------------------------------------------


class StatisticalTestingVerifier(BaseStepVerifier):
    """
    Score: did the agent apply a recognised statistical test and did it run?

    +0.5  a statistical test function is present in code
    +0.2  code produced a result without errors
    +0.3  a p-value or test statistic appears in the output
    """

    step_name = "statistical_testing"

    _STAT_TEST_PATTERNS = re.compile(
        r"\b(ttest_ind|ttest_rel|mannwhitneyu|wilcoxon|kruskal|"
        r"chi2_contingency|fisher_exact|f_oneway|"
        r"pearsonr|spearmanr|kendalltau|"
        r"OLS|logit|LogisticRegression|LinearRegression|"
        r"linregress|polyfit|"
        r"scipy\.stats|statsmodels|sklearn\.linear_model|"
        r"permutation_test|bootstrap)\b",
        re.IGNORECASE,
    )
    _PVALUE_IN_OUTPUT = re.compile(
        r"\b(p.?value|p\s*=\s*[\d.e+-]+|statistic\s*=|t\s*=|F\s*=|chi2\s*=|"
        r"U\s*=|H\s*=|r\s*=\s*[-\d.]+)\b",
        re.IGNORECASE,
    )
    _ERROR_PATTERNS = re.compile(
        r"\b(Error|Traceback|exception)\b", re.IGNORECASE
    )

    def score(self, step_content: str, state: dict[str, Any]) -> float:
        if not step_content:
            return 0.0

        code_blocks = self._extract_code_blocks(step_content)
        code_text = "\n".join(code_blocks) or step_content

        score = 0.0

        # +0.5 for a recognised statistical test
        if self._STAT_TEST_PATTERNS.search(code_text):
            score += 0.5

        # +0.2 if latest tool output has no errors
        tool_results = state.get("tool_results", [])
        if tool_results:
            last_result = str(tool_results[-1])
            if not self._ERROR_PATTERNS.search(last_result):
                score += 0.2
            # +0.3 if output contains p-value / test statistic
            if self._PVALUE_IN_OUTPUT.search(last_result):
                score += 0.3

        return min(1.0, score)


# ---------------------------------------------------------------------------
# Step 5 — Interpretation
# ---------------------------------------------------------------------------


class InterpretationVerifier(BaseStepVerifier):
    """
    Score: did the agent correctly interpret the statistical results?

    +0.4  agent mentions a p-value or significance threshold
    +0.3  agent connects result back to the hypothesis (accept / reject language)
    +0.3  agent provides a directional conclusion (True / False / supported / refuted)
    """

    step_name = "interpretation"

    _PVALUE_MENTION = re.compile(
        r"\b(p.?value|p\s*[<>=]\s*[\d.e+-]+|significant|not significant|"
        r"alpha\s*=|0\.05|0\.01)\b",
        re.IGNORECASE,
    )
    _HYPOTHESIS_CONNECTION = re.compile(
        r"\b(hypothesis|null hypothesis|reject|fail to reject|accept|"
        r"support|refute|confirm|disprove|consistent with|inconsistent)\b",
        re.IGNORECASE,
    )
    _CONCLUSION_DIRECTION = re.compile(
        r"\b(true|false|yes|no|supported|refuted|confirmed|disproved|"
        r"positive|negative|significant difference|no significant)\b",
        re.IGNORECASE,
    )

    def score(self, step_content: str, state: dict[str, Any]) -> float:
        if not step_content:
            return 0.0

        score = 0.0
        if self._PVALUE_MENTION.search(step_content):
            score += 0.4
        if self._HYPOTHESIS_CONNECTION.search(step_content):
            score += 0.3
        if self._CONCLUSION_DIRECTION.search(step_content):
            score += 0.3

        return min(1.0, score)


# ---------------------------------------------------------------------------
# Step 6 — Answer submission
# ---------------------------------------------------------------------------


class AnswerSubmissionVerifier(BaseStepVerifier):
    """
    Score: did the agent submit the correct True/False verdict?

    +0.4  agent calls submit_answer (format compliance)
    +0.6  submitted answer matches ground truth (correctness)
    """

    step_name = "answer_submission"

    _SUBMIT_CALL = re.compile(r"\bsubmit_answer\b", re.IGNORECASE)
    _TRUE_FALSE = re.compile(r"\b(true|false)\b", re.IGNORECASE)

    def score(self, step_content: str, state: dict[str, Any]) -> float:
        submitted_answer = state.get("submitted_answer")
        ground_truth = state.get("ground_truth", "")

        score = 0.0

        # +0.4 for format compliance (submit_answer called)
        if self._SUBMIT_CALL.search(step_content) or submitted_answer is not None:
            score += 0.4

        # +0.6 for correctness
        if submitted_answer is not None and ground_truth:
            pred = str(submitted_answer).strip().lower()
            gt = str(ground_truth).strip().lower()
            # Normalise to boolean
            pred_bool = pred in {"true", "1", "yes", "supported"}
            gt_bool = gt in {"true", "1", "yes", "supported"}
            if pred_bool == gt_bool:
                score += 0.6

        return min(1.0, score)
