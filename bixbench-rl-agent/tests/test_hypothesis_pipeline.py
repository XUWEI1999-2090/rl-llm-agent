"""
Tests for the hypothesis training pipeline:
  - NemotronHypothesisDataset schema and API
  - AnalysisProtocol weight invariants
  - Step verifiers basic scoring
  - StepRubric trajectory scoring
  - CrowRLEnv env structure
  - train_grpo_hypothesis.py integration checks
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_nemotron_dataset_importable():
    source = (SRC / "dataset/nemotron_dataset.py").read_text()
    assert "NemotronHypothesisDataset" in source
    assert "HypothesisSample" in source


def test_hypothesis_sample_answer_bool():
    from src.dataset.nemotron_dataset import HypothesisSample

    s = HypothesisSample(capsule_id="c1", hypothesis="X", answer="True")
    assert s.answer_bool is True

    s2 = HypothesisSample(capsule_id="c2", hypothesis="X", answer="False")
    assert s2.answer_bool is False

    s3 = HypothesisSample(capsule_id="c3", hypothesis="X", answer="supported")
    assert s3.answer_bool is True


def test_hypothesis_sample_format_prompt():
    from src.dataset.nemotron_dataset import HypothesisSample

    s = HypothesisSample(
        capsule_id="cap1",
        hypothesis="Gene A expression is higher in treated cells.",
        answer="True",
        description="A bioinformatics study.",
    )
    prompt = s.format_prompt()
    assert "Hypothesis" in prompt
    assert "Gene A" in prompt
    assert "submit_answer" in prompt


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_protocol_weights_sum_to_one():
    from src.verifiers.protocol import HYPOTHESIS_PROTOCOL

    total = sum(s.weight for s in HYPOTHESIS_PROTOCOL)
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


def test_protocol_has_six_steps():
    from src.verifiers.protocol import HYPOTHESIS_PROTOCOL

    assert len(HYPOTHESIS_PROTOCOL) == 6


def test_protocol_step_names():
    from src.verifiers.protocol import HYPOTHESIS_PROTOCOL

    expected = {
        "hypothesis_understanding",
        "data_loading",
        "exploratory_analysis",
        "statistical_testing",
        "interpretation",
        "answer_submission",
    }
    assert set(HYPOTHESIS_PROTOCOL.step_names()) == expected


def test_protocol_invalid_weights_raise():
    from src.verifiers.protocol import AnalysisProtocol, ProtocolStep
    import pytest

    with pytest.raises(ValueError, match="sum to 1.0"):
        AnalysisProtocol(
            steps=[
                ProtocolStep(name="a", description="x", weight=0.5),
                ProtocolStep(name="b", description="y", weight=0.3),
            ]
        )


# ---------------------------------------------------------------------------
# Step verifiers
# ---------------------------------------------------------------------------


def test_hypothesis_understanding_zero_for_empty():
    from src.verifiers.step_verifiers import HypothesisUnderstandingVerifier

    v = HypothesisUnderstandingVerifier()
    assert v.score("", {}) == 0.0


def test_hypothesis_understanding_nonzero_for_relevant_text():
    from src.verifiers.step_verifiers import HypothesisUnderstandingVerifier

    v = HypothesisUnderstandingVerifier()
    state = {"hypothesis": "Gene expression levels differ between treated and control cells"}
    content = (
        "The hypothesis states that gene expression levels differ between treated "
        "and control cells. I need to test whether there is a significant difference."
    )
    score = v.score(content, state)
    assert score > 0.3


def test_data_loading_detects_pd_read_csv():
    from src.verifiers.step_verifiers import DataLoadingVerifier

    v = DataLoadingVerifier()
    content = "```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```"
    score = v.score(content, {"tool_results": ["   A  B\n0  1  2"]})
    assert score >= 0.6


def test_exploratory_analysis_multiple_techniques():
    from src.verifiers.step_verifiers import ExploratoryAnalysisVerifier

    v = ExploratoryAnalysisVerifier()
    content = """
```python
print(df.head())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
```
"""
    score = v.score(content, {})
    assert score >= 0.5


def test_statistical_testing_detects_scipy():
    from src.verifiers.step_verifiers import StatisticalTestingVerifier

    v = StatisticalTestingVerifier()
    content = """
```python
from scipy import stats
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t={t_stat:.3f}, p_value={p_value:.4f}")
```
"""
    tool_result = "t=2.341, p_value=0.0213"
    score = v.score(content, {"tool_results": [tool_result]})
    assert score >= 0.7


def test_interpretation_full_score():
    from src.verifiers.step_verifiers import InterpretationVerifier

    v = InterpretationVerifier()
    content = (
        "The p-value is 0.021 (< 0.05), indicating a significant difference. "
        "We reject the null hypothesis. The hypothesis is supported by the data."
    )
    score = v.score(content, {})
    assert score == 1.0


def test_answer_submission_correct():
    from src.verifiers.step_verifiers import AnswerSubmissionVerifier

    v = AnswerSubmissionVerifier()
    score = v.score(
        "submit_answer('True')",
        {"submitted_answer": "True", "ground_truth": "True"},
    )
    assert score == 1.0


def test_answer_submission_incorrect():
    from src.verifiers.step_verifiers import AnswerSubmissionVerifier

    v = AnswerSubmissionVerifier()
    score = v.score(
        "submit_answer('False')",
        {"submitted_answer": "False", "ground_truth": "True"},
    )
    # format correct (+0.4), answer wrong (no +0.6)
    assert math.isclose(score, 0.4, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# StepRubric
# ---------------------------------------------------------------------------


def test_step_rubric_scores_trajectory():
    from src.verifiers.rubric import StepRubric
    from src.verifiers.protocol import HYPOTHESIS_PROTOCOL

    rubric = StepRubric(protocol=HYPOTHESIS_PROTOCOL)
    traj = {
        "hypothesis": "Gene X is upregulated in cancer cells",
        "ground_truth": "True",
        "submitted_answer": "True",
        "steps": [
            {
                "step": 1,
                "action": (
                    "The hypothesis says gene X is upregulated. "
                    "I need to test whether expression levels differ."
                ),
                "tool_result": "",
            },
            {
                "step": 2,
                "action": "```python\ndf = pd.read_csv('expr.csv')\n```",
                "tool_result": "   gene  expr\n0  GeneX  5.2",
            },
            {
                "step": 3,
                "action": "```python\nprint(df.head())\nprint(df.describe())\n```",
                "tool_result": "count  100\nmean   3.5",
            },
            {
                "step": 4,
                "action": (
                    "```python\nfrom scipy import stats\n"
                    "t, pvalue = stats.ttest_ind(cancer, normal)\n"
                    "print(f'p_value={pvalue}')\n```"
                ),
                "tool_result": "p_value=0.003",
            },
            {
                "step": 5,
                "action": (
                    "The p-value is 0.003 < 0.05, significant. "
                    "We reject the null hypothesis. The hypothesis is supported."
                ),
                "tool_result": "",
            },
            {
                "step": 6,
                "action": "submit_answer('True')",
                "tool_result": "Submitted answer: True",
            },
        ],
    }

    step_scores = rubric.score_trajectory(traj)
    total = rubric.aggregate(step_scores)

    assert set(step_scores.keys()) == set(HYPOTHESIS_PROTOCOL.step_names())
    assert all(0.0 <= v <= 1.0 for v in step_scores.values())
    assert 0.0 <= total <= 1.0
    assert total > 0.3, f"Expected reasonable total reward, got {total}"


def test_step_rubric_reward_func_compatible():
    from src.verifiers.rubric import StepRubric

    rubric = StepRubric()
    reward = rubric.reward_func(
        prompt="Does gene X differ?",
        completion=(
            "The p-value is 0.01, significant. "
            "We reject the null. submit_answer('True')"
        ),
        answer="True",
        state={"submitted_answer": "True", "ground_truth": "True"},
    )
    assert isinstance(reward, float)
    assert 0.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# CrowRLEnv
# ---------------------------------------------------------------------------


def test_crow_env_importable():
    source = (SRC / "envs/crow_env.py").read_text()
    assert "CrowRLEnv" in source
    assert "CrowDataset" in source
    assert "run_crow_episode" in source
    assert "collect_crow_rollouts" in source


def test_crow_env_uses_data_analysis_crow():
    source = (SRC / "envs/crow_env.py").read_text()
    assert "DataAnalysisEnv" in source
    assert "fhda" in source


def test_crow_env_hypothesis_system_prompt():
    source = (SRC / "envs/crow_env.py").read_text()
    assert "HYPOTHESIS_SYSTEM_PROMPT" in source
    assert "submit_answer" in source


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------


def test_train_script_importable():
    source = (ROOT / "scripts/train_grpo_hypothesis.py").read_text()
    assert "StepLevelGRPOGrouper" in source
    assert "build_training_samples" in source
    assert "collect_crow_rollouts" in source
    assert "run_verifiers_training" in source


def test_train_script_uses_nemotron_dataset():
    source = (ROOT / "scripts/train_grpo_hypothesis.py").read_text()
    assert "NemotronHypothesisDataset" in source


def test_train_script_uses_step_rubric():
    source = (ROOT / "scripts/train_grpo_hypothesis.py").read_text()
    assert "StepRubric" in source
    assert "HYPOTHESIS_PROTOCOL" in source


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_hypothesis_config_exists():
    cfg_path = ROOT / "configs/grpo_hypothesis.yaml"
    assert cfg_path.exists()


def test_hypothesis_config_valid_yaml():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/grpo_hypothesis.yaml").read_text())
    assert cfg["dataset"]["hf_id"] == "nvidia/Nemotron-RL-bixbench_hypothesis"
    assert cfg["grpo"]["use_step_level"] is True
    assert cfg["env"]["name"] == "CrowRLEnv"


def test_hypothesis_config_protocol_weights_sum():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/grpo_hypothesis.yaml").read_text())
    steps = cfg["protocol"]["steps"]
    total = sum(s["weight"] for s in steps)
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# setup_env.sh
# ---------------------------------------------------------------------------


def test_setup_installs_data_analysis_crow():
    setup = (ROOT / "scripts/setup_env.sh").read_text()
    assert "data-analysis-crow" in setup
    assert "Future-House/data-analysis-crow" in setup


def test_setup_installs_verifiers():
    setup = (ROOT / "scripts/setup_env.sh").read_text()
    assert "verifiers" in setup
