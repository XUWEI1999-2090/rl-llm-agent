"""
src/dataset/nemotron_dataset.py
================================
Loader for the nvidia/Nemotron-RL-bixbench_hypothesis HuggingFace dataset.

Dataset schema inferred from BixBench hypothesis mode and NVIDIA NeMo RL paper:
  - hypothesis   : str  — The biological hypothesis to test
  - answer       : str  — Ground-truth verdict ("True" or "False")
  - capsule_id   : str  — Unique identifier of the data capsule
  - data_path    : str | None — Relative path to capsule data files (optional)
  - description  : str | None — Extended problem description (optional)
  - split        : str  — "train" / "validation" / "test"

Reference: https://huggingface.co/datasets/nvidia/Nemotron-RL-bixbench_hypothesis
"""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_HF_DATASET_ID = "nvidia/Nemotron-RL-bixbench_hypothesis"
_LOCAL_CAPSULE_REPO = "baseline-envs/data-analysis/v3.1"


@dataclass
class HypothesisSample:
    """One hypothesis-testing problem from the Nemotron dataset."""

    capsule_id: str
    hypothesis: str
    answer: str  # "True" or "False"
    description: str = ""
    data_path: str | None = None
    metadata: dict = field(default_factory=dict)
    split: str = "train"

    @property
    def answer_bool(self) -> bool:
        return self.answer.strip().lower() in {"true", "1", "yes", "supported"}

    def format_prompt(self) -> str:
        """Return a structured prompt for the agent."""
        parts = [
            "# Biological Hypothesis Evaluation Task\n",
            f"## Hypothesis\n{self.hypothesis}\n",
        ]
        if self.description:
            parts.append(f"## Background\n{self.description}\n")
        parts += [
            "## Instructions\n"
            "Analyse the available data to determine whether the hypothesis is "
            "**supported** or **refuted** by the data.\n\n"
            "Follow this protocol:\n"
            "1. Load and inspect the relevant data files.\n"
            "2. Perform exploratory data analysis (distributions, shapes, NA counts).\n"
            "3. Apply the appropriate statistical test(s) to evaluate the hypothesis.\n"
            "4. Interpret the results (p-values, effect sizes, confidence intervals).\n"
            "5. Call `submit_answer` with your verdict: \"True\" (supported) "
            "or \"False\" (refuted).\n",
        ]
        return "\n".join(parts)


class NemotronHypothesisDataset:
    """
    Interface to the nvidia/Nemotron-RL-bixbench_hypothesis dataset.

    Usage (requires `datasets` and HuggingFace authentication):

        ds = NemotronHypothesisDataset(split="train")
        sample = ds[0]
        print(sample.hypothesis, sample.answer)

    Local capsule data is expected in `capsule_data_dir`.  When running the
    data-analysis-crow environment the capsules are sourced from the hosted
    DataRepo (see CapsuleDataset in fhda.dataset).
    """

    def __init__(
        self,
        split: str = "train",
        capsule_data_dir: str | Path | None = None,
        streaming: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.split = split
        self.capsule_data_dir = Path(capsule_data_dir) if capsule_data_dir else None
        self._samples: list[HypothesisSample] = []
        self._hf_dataset = None

        self._load(streaming=streaming, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, streaming: bool, cache_dir: str | Path | None) -> None:
        try:
            from datasets import load_dataset

            logger.info("Loading %s (split=%s) from HuggingFace...", _HF_DATASET_ID, self.split)
            hf_ds = load_dataset(
                _HF_DATASET_ID,
                split=self.split,
                streaming=streaming,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=True,
            )
            self._hf_dataset = hf_ds
            if not streaming:
                self._samples = [self._row_to_sample(row) for row in hf_ds]
                logger.info("Loaded %d samples (split=%s)", len(self._samples), self.split)
        except Exception as exc:
            logger.warning(
                "Could not load %s from HuggingFace (%s). "
                "Dataset will be empty — ensure you have authenticated with "
                "`huggingface-cli login` and the package `datasets` is installed.",
                _HF_DATASET_ID,
                exc,
            )
            self._samples = []

    @staticmethod
    def _row_to_sample(row: dict) -> HypothesisSample:
        """Convert a raw HuggingFace row to a HypothesisSample.

        The field names below cover the schema variants seen in the dataset card
        and the related BixBench/data-analysis-crow code.
        """
        return HypothesisSample(
            capsule_id=str(
                row.get("capsule_id") or row.get("id") or row.get("problem_id", "unknown")
            ),
            hypothesis=str(
                row.get("hypothesis") or row.get("question") or row.get("problem", "")
            ),
            answer=str(row.get("answer", "False")).strip(),
            description=str(row.get("description") or row.get("background", "")),
            data_path=row.get("data_path") or row.get("dataset_path"),
            metadata={k: v for k, v in row.items()},
            split=row.get("split", "train"),
        )

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> HypothesisSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[HypothesisSample]:
        return iter(self._samples)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def as_hf_dataset(self):
        """Return the underlying HuggingFace Dataset (or None if not loaded)."""
        return self._hf_dataset

    def get_capsule_dir(self, sample: HypothesisSample) -> Path | None:
        """Return the local directory for the capsule's data files, if available."""
        if self.capsule_data_dir is None:
            return None
        candidate = self.capsule_data_dir / f"CapsuleFolder-{sample.capsule_id}"
        if candidate.exists():
            return candidate
        # Fallback: search for any directory containing the capsule_id
        for d in self.capsule_data_dir.iterdir():
            if d.is_dir() and sample.capsule_id in d.name:
                return d
        return None

    def prepare_work_dir(self, sample: HypothesisSample, base_tmp: Path | None = None) -> Path:
        """Copy capsule data files to a fresh temporary directory and return it.

        This mirrors what CapsuleDataset.get_new_env_by_idx() does in fhda.
        """
        tmp = Path(tempfile.mkdtemp(dir=base_tmp))
        capsule_dir = self.get_capsule_dir(sample)
        if capsule_dir is not None:
            for item in capsule_dir.iterdir():
                if item.suffix in {".ipynb"} or item.name in {"metadata.json", "checksum"}:
                    continue
                if item.is_dir():
                    shutil.copytree(item, tmp / item.name)
                else:
                    shutil.copy(item, tmp)
        return tmp

    def train_val_split(
        self,
        val_frac: float = 0.2,
        seed: int = 42,
    ) -> tuple["NemotronHypothesisDataset", "SubsetNemotronDataset"]:
        """Split this dataset into a train subset and a validation subset.

        Returns a *(train_ds, val_ds)* tuple where both objects expose the
        same ``__len__`` / ``__getitem__`` / ``prepare_work_dir`` interface as
        :class:`NemotronHypothesisDataset`.

        This is used when the HuggingFace dataset only provides a ``train``
        split (e.g. ``nvidia/Nemotron-RL-bixbench_hypothesis``).  The split
        is deterministic given *seed* so that repeated runs are reproducible.

        Args:
            val_frac: Fraction of samples reserved for validation (default 0.2).
            seed: Random seed for reproducibility (default 42).

        Returns:
            (train_subset, val_subset) — both are :class:`SubsetNemotronDataset`.
        """
        n = len(self)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_val = max(1, int(val_frac * n))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        logger.info(
            "train_val_split: %d total → %d train, %d val (val_frac=%.2f, seed=%d)",
            n,
            len(train_idx),
            len(val_idx),
            val_frac,
            seed,
        )
        return SubsetNemotronDataset(self, train_idx), SubsetNemotronDataset(self, val_idx)


class SubsetNemotronDataset:
    """A deterministic subset of a :class:`NemotronHypothesisDataset`.

    Wraps a list of integer indices into the parent dataset so that
    train/validation subsets share the same underlying data and capsule
    directory without duplicating samples in memory.

    Exposes the same interface as :class:`NemotronHypothesisDataset`:
    ``__len__``, ``__getitem__``, ``__iter__``, ``prepare_work_dir``, and
    ``capsule_data_dir``.
    """

    def __init__(
        self,
        base: "NemotronHypothesisDataset",
        indices: list[int],
    ) -> None:
        self._base = base
        self._indices = indices
        # Expose capsule_data_dir so CrowDataset / callers can access it.
        self.capsule_data_dir = base.capsule_data_dir

    # ------------------------------------------------------------------
    # Sequence interface (mirrors NemotronHypothesisDataset)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> HypothesisSample:
        return self._base[self._indices[i]]

    def __iter__(self) -> Iterator[HypothesisSample]:
        return (self._base[i] for i in self._indices)

    # ------------------------------------------------------------------
    # Helpers (delegate to base)
    # ------------------------------------------------------------------

    def prepare_work_dir(
        self, sample: HypothesisSample, base_tmp: Path | None = None
    ) -> Path:
        """Delegate to the parent dataset's ``prepare_work_dir``."""
        return self._base.prepare_work_dir(sample, base_tmp=base_tmp)
