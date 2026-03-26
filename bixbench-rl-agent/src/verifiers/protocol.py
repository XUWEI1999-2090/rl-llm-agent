"""
src/verifiers/protocol.py
==========================
AnalysisProtocol — defines the ordered steps an agent must follow for
hypothesis-testing analysis tasks, together with the scoring weight of
each step.

The protocol is the central contract between the environment and the
reward system.  Every step has:

  name        — machine-readable identifier
  description — human-readable explanation
  weight      — fraction of the total reward attributed to this step
                (all weights must sum to 1.0)
  required    — whether the step is mandatory (missing it = 0 for that step)

The default HYPOTHESIS_PROTOCOL mirrors the analysis workflow recommended by
the data-analysis-crow system prompt:
  1.  Hypothesis understanding   (0.10)
  2.  Data loading               (0.15)
  3.  Exploratory data analysis  (0.20)
  4.  Statistical testing        (0.25)
  5.  Result interpretation      (0.15)
  6.  Answer submission          (0.15)
                                ------
                                 1.00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class ProtocolStep:
    """Specification of a single protocol step."""

    name: str
    description: str
    weight: float
    required: bool = True
    keywords: tuple[str, ...] = field(default_factory=tuple)  # hint patterns for verifiers

    def __post_init__(self) -> None:
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Step weight must be in [0, 1], got {self.weight!r}")


@dataclass
class AnalysisProtocol:
    """
    Ordered sequence of ProtocolSteps that defines the expected analysis
    workflow and the scoring rubric.

    Attributes
    ----------
    steps   : Ordered list of protocol steps.
    name    : Human-readable name for this protocol.
    """

    steps: list[ProtocolStep]
    name: str = "analysis_protocol"

    def __post_init__(self) -> None:
        total = sum(s.weight for s in self.steps)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ProtocolStep weights must sum to 1.0, got {total:.6f} "
                f"for protocol '{self.name}'"
            )

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> ProtocolStep:
        return self.steps[idx]

    def __iter__(self):
        return iter(self.steps)

    def step_by_name(self, name: str) -> ProtocolStep | None:
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def max_score(self) -> float:
        """Returns 1.0 — present for symmetry with partial_score()."""
        return 1.0

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]


# ---------------------------------------------------------------------------
# Default protocol for hypothesis-testing tasks (BixBench / Nemotron dataset)
# ---------------------------------------------------------------------------

HYPOTHESIS_PROTOCOL = AnalysisProtocol(
    name="hypothesis_testing",
    steps=[
        ProtocolStep(
            name="hypothesis_understanding",
            description=(
                "Agent correctly identifies the hypothesis, the biological "
                "entities involved, and the expected direction of the effect."
            ),
            weight=0.10,
            required=True,
            keywords=(
                "hypothesis",
                "relationship",
                "variable",
                "measure",
                "expect",
                "test whether",
            ),
        ),
        ProtocolStep(
            name="data_loading",
            description=(
                "Agent loads the relevant data files from the work directory "
                "using pandas, CSV readers, or equivalent."
            ),
            weight=0.15,
            required=True,
            keywords=(
                "read_csv",
                "read_excel",
                "read_table",
                "pd.read",
                "load_dataset",
                "open(",
                "list_workdir",
            ),
        ),
        ProtocolStep(
            name="exploratory_analysis",
            description=(
                "Agent examines the structure, distributions, and quality of "
                "the data (shapes, dtypes, missing values, summary statistics)."
            ),
            weight=0.20,
            required=True,
            keywords=(
                ".head(",
                ".tail(",
                ".describe(",
                ".info(",
                ".shape",
                ".dtypes",
                ".value_counts(",
                ".isnull(",
                ".isna(",
                "hist(",
                "boxplot(",
            ),
        ),
        ProtocolStep(
            name="statistical_testing",
            description=(
                "Agent applies an appropriate statistical test (t-test, "
                "Mann-Whitney, chi-square, ANOVA, correlation, regression, …) "
                "and the code executes without errors."
            ),
            weight=0.25,
            required=True,
            keywords=(
                "scipy.stats",
                "statsmodels",
                "ttest",
                "mannwhitneyu",
                "wilcoxon",
                "chi2_contingency",
                "pearsonr",
                "spearmanr",
                "anova",
                "kruskal",
                "logistic",
                "linregress",
                "pvalue",
                "p_value",
                "p-value",
            ),
        ),
        ProtocolStep(
            name="interpretation",
            description=(
                "Agent interprets statistical results in the context of the "
                "hypothesis (p-value, effect size, confidence interval, "
                "direction of effect)."
            ),
            weight=0.15,
            required=True,
            keywords=(
                "p-value",
                "p_value",
                "pvalue",
                "p <",
                "p=",
                "significant",
                "not significant",
                "effect size",
                "confidence interval",
                "therefore",
                "conclude",
                "supported",
                "refuted",
                "reject",
                "fail to reject",
            ),
        ),
        ProtocolStep(
            name="answer_submission",
            description=(
                "Agent submits a final True/False verdict via submit_answer "
                "that matches the ground-truth answer."
            ),
            weight=0.15,
            required=True,
            keywords=(
                "submit_answer",
                "True",
                "False",
            ),
        ),
    ],
)
