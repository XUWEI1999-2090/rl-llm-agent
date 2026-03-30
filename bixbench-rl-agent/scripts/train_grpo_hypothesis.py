"""
scripts/train_grpo_hypothesis.py
==================================
GRPO training on the nvidia/Nemotron-RL-bixbench_hypothesis dataset.

Pipeline
--------
1. Load dataset  : nvidia/Nemotron-RL-bixbench_hypothesis (HuggingFace)
2. Environment   : data-analysis-crow (CrowRLEnv via DataAnalysisEnv)
3. Rewards       : AnalysisProtocol + StepRubric (per-step verifiers)
4. Training      : Step-level GRPO via StepLevelGRPOGrouper

Step-level GRPO (Edison Innovation #2)
---------------------------------------
Standard GRPO groups G parallel rollouts of the same *full* trajectory and
normalises rewards across them.  Step-level GRPO instead groups G parallel
executions of the *same step* across trajectories and normalises at that
granularity, giving a much denser reward signal.

The StepRubric scores each protocol step independently, so for every step t
we have G rewards { r_t^1, …, r_t^G }.  The advantage at step t is:

    A_t^i = (r_t^i - mean(r_t)) / std(r_t)

Run modes
---------
  A.  This script (direct / debugging):
        python scripts/train_grpo_hypothesis.py --config configs/grpo_hypothesis.yaml

  B.  NeMo RL (full multi-GPU):
        ng_run +config_paths=[configs/grpo_hypothesis.yaml] ...

  C.  verifiers + GRPOTrainer (single/multi-GPU via accelerate):
        accelerate launch scripts/train_grpo_hypothesis.py \
            --config configs/grpo_hypothesis.yaml \
            --use-verifiers-trainer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step-level GRPO grouper (extended to handle StepRubric scores)
# ---------------------------------------------------------------------------


class StepLevelGRPOGrouper:
    """
    Extend the base StepLevelGRPOGrouper to work with StepRubric per-step scores.

    For each protocol step, we group G parallel rollout scores and compute the
    per-rollout advantage via mean/std normalisation.

    This implements the key innovation from:
      Edison Scientific / NVIDIA — Step-level GRPO for multi-step agents
    """

    def __init__(self, group_size: int = 8, min_reward_std: float = 1e-8) -> None:
        self.group_size = group_size
        self.min_reward_std = min_reward_std

    def build_training_samples(
        self,
        trajectories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Build step-level training samples from parallel trajectories.

        For each capsule, trajectories are grouped by (capsule_id, step_name).
        Within each group, advantages are computed as normalised step scores.

        Returns list of dicts with keys:
          step_name, capsule_id, action, advantage, log_prob_old,
          step_score, protocol_step_index
        """
        # Group by (capsule_id, step_name)
        by_capsule_step: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for traj in trajectories:
            capsule_id = traj.get("capsule_id", "unknown")
            step_scores = traj.get("step_scores", {})
            steps = traj.get("steps", [])

            for proto_step_name, score in step_scores.items():
                # Find the most relevant conversation step for this protocol step
                # (we use the first step with relevant content as the "action")
                action_text = _find_action_for_protocol_step(steps, proto_step_name)
                log_prob = _find_logprob_for_protocol_step(steps, proto_step_name)

                by_capsule_step[(capsule_id, proto_step_name)].append(
                    {
                        "step_name": proto_step_name,
                        "capsule_id": capsule_id,
                        "action": action_text,
                        "step_score": score,
                        "log_prob": log_prob,
                    }
                )

        samples: list[dict[str, Any]] = []
        skipped = 0

        for (capsule_id, step_name), group in by_capsule_step.items():
            if len(group) < 2:
                skipped += 1
                continue

            scores = [g["step_score"] for g in group]
            mean_r = sum(scores) / len(scores)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in scores) / len(scores))

            if std_r < self.min_reward_std:
                skipped += 1
                continue

            advantages = [(r - mean_r) / (std_r + self.min_reward_std) for r in scores]

            for item, adv in zip(group, advantages):
                if item["log_prob"] is None:
                    continue
                samples.append(
                    {
                        "step_name": item["step_name"],
                        "capsule_id": item["capsule_id"],
                        "action": item["action"],
                        "advantage": adv,
                        "log_prob_old": item["log_prob"],
                        "step_score": item["step_score"],
                    }
                )

        logger.info(
            "Built %d training samples from %d trajectories (%d groups skipped)",
            len(samples),
            len(trajectories),
            skipped,
        )
        return samples

    @staticmethod
    def compute_grpo_loss(
        samples: list[dict[str, Any]],
        model_logprobs: list[float],
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
    ) -> float:
        """
        PPO-clip GRPO loss over step-level samples.

        True training is handled by NeMo RL / verifiers GRPOTrainer;
        this is provided for reference and offline debugging.
        """
        total_loss = 0.0
        for sample, new_lp in zip(samples, model_logprobs):
            old_lp = sample["log_prob_old"]
            adv = sample["advantage"]
            ratio = math.exp(new_lp - old_lp)
            clipped = max(1 - clip_eps, min(1 + clip_eps, ratio))
            policy_loss = -min(ratio * adv, clipped * adv)
            kl = old_lp - new_lp
            total_loss += policy_loss + kl_coef * kl
        return total_loss / len(samples) if samples else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def train(config: dict) -> None:
    """Async training loop for step-level GRPO on hypothesis tasks."""
    from src.dataset.nemotron_dataset import NemotronHypothesisDataset
    from src.envs.crow_env import CrowDataset, collect_crow_rollouts
    from src.verifiers.protocol import HYPOTHESIS_PROTOCOL
    from src.verifiers.rubric import StepRubric
    from src.agents.notebook_agent import make_bixbench_agent

    log_dir = Path(config.get("logging", {}).get("log_dir", "logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(config.get("training", {}).get("checkpoint_dir", "checkpoints/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_cfg = config.get("dataset", {})
    capsule_data_dir = ds_cfg.get("capsule_data_dir")
    val_fraction = float(ds_cfg.get("val_fraction", 0.2))
    split_seed = int(ds_cfg.get("split_seed", 42))

    # Load only the train split — the HF dataset may not have a 'validation'
    # split (e.g. on Colab).  We carve out a deterministic val subset instead.
    full_ds = NemotronHypothesisDataset(
        split=ds_cfg.get("train_split", "train"),
        capsule_data_dir=capsule_data_dir,
    )
    train_ds, val_ds = full_ds.split_train_val(
        val_fraction=val_fraction, seed=split_seed
    )
    logger.info(
        "Dataset: %d train, %d val samples (split from train, val_fraction=%.2f)",
        len(train_ds), len(val_ds), val_fraction,
    )

    # ── Environment ──────────────────────────────────────────────────────────
    use_docker = config.get("env", {}).get("use_docker", True)
    docker_image = config.get("env", {}).get("docker_image", "futurehouse/bixbench:aviary-notebook-env")
    train_crow = CrowDataset(train_ds, use_docker=use_docker, docker_image=docker_image)
    val_crow = CrowDataset(val_ds, use_docker=use_docker, docker_image=docker_image)

    # ── Reward rubric ─────────────────────────────────────────────────────────
    rubric = StepRubric(protocol=HYPOTHESIS_PROTOCOL)

    # ── Rollout agent ─────────────────────────────────────────────────────────
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "gpt-4o")
    # LiteLLM requires a provider prefix (e.g. "openai/gpt-4o-mini").
    # If the name contains no "/" we assume it is an OpenAI-compatible model
    # and prepend "openai/" automatically — this covers Colab / chatanywhere
    # endpoints where the user just writes the bare model name.
    if "/" not in model_name:
        model_name = f"openai/{model_name}"
        logger.info("Model name normalized to '%s' for LiteLLM provider routing", model_name)
    agent = make_bixbench_agent(
        model_name=model_name,
        model_base_url=model_cfg.get("base_url"),
        temperature=config.get("grpo", {}).get("temperature", 0.7),
    )

    # ── GRPO grouper ─────────────────────────────────────────────────────────
    grpo_cfg = config.get("grpo", {})
    grouper = StepLevelGRPOGrouper(
        group_size=grpo_cfg.get("group_size", 8),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_cfg = config.get("training", {})
    n_iters = train_cfg.get("n_iterations", 200)
    n_samples = train_cfg.get("samples_per_iter", 4)
    n_parallel = train_cfg.get("n_parallel_rollouts", 8)
    eval_every = train_cfg.get("eval_every", 20)

    logger.info(
        "Training: %d iterations, %d samples × %d rollouts = %d rollouts/iter",
        n_iters,
        n_samples,
        n_parallel,
        n_samples * n_parallel,
    )

    metrics_path = log_dir / "training_metrics.jsonl"
    with open(metrics_path, "w") as metrics_f:
        for iteration in range(1, n_iters + 1):
            t0 = time.time()

            # 1. Collect rollouts
            trajectories = await collect_crow_rollouts(
                train_crow,
                agent,
                rubric,
                n_samples=n_samples,
                n_parallel=n_parallel,
                max_steps=config.get("env", {}).get("max_steps", 30),
            )

            if not trajectories:
                logger.warning("Iter %d: no successful rollouts, skipping.", iteration)
                continue

            # 2. Step-level GRPO grouping
            samples = grouper.build_training_samples(trajectories)

            # 3. Model update
            #    Real training is handled by NeMo RL (ng_run) or
            #    verifiers.GRPOTrainer.  The advantage tensors produced above
            #    are fed into that trainer via the reward_func interface.
            #    See configs/grpo_hypothesis.yaml for NeMo RL config.

            mean_reward = sum(t["total_reward"] for t in trajectories) / len(trajectories)
            step_score_means = _compute_mean_step_scores(trajectories)

            entry = {
                "iteration": iteration,
                "mean_reward": mean_reward,
                "n_trajectories": len(trajectories),
                "n_grpo_samples": len(samples),
                "step_scores": step_score_means,
                "elapsed_s": time.time() - t0,
            }
            metrics_f.write(json.dumps(entry) + "\n")
            metrics_f.flush()

            logger.info(
                "Iter %d/%d | reward=%.3f | traj=%d | grpo_samples=%d | %.1fs",
                iteration,
                n_iters,
                mean_reward,
                len(trajectories),
                len(samples),
                entry["elapsed_s"],
            )
            if logger.isEnabledFor(logging.DEBUG):
                for step_name, score in step_score_means.items():
                    logger.debug("  [step] %s = %.3f", step_name, score)

            # 4. Periodic evaluation
            if iteration % eval_every == 0 and len(val_crow) > 0:
                val_trajs = await collect_crow_rollouts(
                    val_crow,
                    agent,
                    rubric,
                    n_samples=min(5, len(val_crow)),
                    n_parallel=3,
                    max_steps=config.get("env", {}).get("max_steps", 30),
                )
                if val_trajs:
                    val_reward = sum(t["total_reward"] for t in val_trajs) / len(val_trajs)
                    logger.info("  [Eval] val_reward=%.3f", val_reward)
                    entry["val_reward"] = val_reward
                    ckpt = ckpt_dir / f"checkpoint_iter{iteration:04d}.json"
                    ckpt.write_text(json.dumps(entry, indent=2))

    logger.info("Training complete. Metrics saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# verifiers GRPOTrainer path (optional — for direct GPU training)
# ---------------------------------------------------------------------------


def build_verifiers_env(config: dict):
    """
    Build a verifiers.MultiTurnEnv + Rubric for use with vf.GRPOTrainer.

    This enables training with:
        CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model <model>
        CUDA_VISIBLE_DEVICES=2,3 accelerate launch scripts/train_grpo_hypothesis.py \
            --config configs/grpo_hypothesis.yaml --use-verifiers-trainer

    The verifiers GRPOTrainer handles:
      - Async batched rollout collection
      - Advantage normalisation (trajectory-level by default)
      - PPO-clip gradient updates
    """
    try:
        import verifiers as vf
        from datasets import Dataset as HFDataset
    except ImportError as exc:
        raise ImportError(
            "The `verifiers` package is required for this mode. "
            "Install with: pip install 'verifiers[all]'"
        ) from exc

    from src.dataset.nemotron_dataset import NemotronHypothesisDataset
    from src.verifiers.protocol import HYPOTHESIS_PROTOCOL
    from src.verifiers.rubric import StepRubric

    rubric_obj = StepRubric(protocol=HYPOTHESIS_PROTOCOL)
    ds_cfg = config.get("dataset", {})
    capsule_data_dir = ds_cfg.get("capsule_data_dir")

    # Load only the train split and carve out val — same logic as train() so
    # that --use-verifiers-trainer doesn't attempt to load the missing
    # 'validation' split on Colab / HF datasets that only have 'train'.
    full_ds = NemotronHypothesisDataset(
        split=ds_cfg.get("train_split", "train"),
        capsule_data_dir=capsule_data_dir,
    )
    val_fraction = float(ds_cfg.get("val_fraction", 0.2))
    split_seed = int(ds_cfg.get("split_seed", 42))
    train_ds, val_ds = full_ds.split_train_val(
        val_fraction=val_fraction, seed=split_seed
    )

    def to_hf(ds: NemotronHypothesisDataset) -> HFDataset:
        return HFDataset.from_list(
            [
                {
                    "question": s.format_prompt(),
                    "answer": s.answer,
                    "hypothesis": s.hypothesis,
                    "capsule_id": s.capsule_id,
                }
                for s in ds
            ]
        )

    parser = vf.XMLParser(["think", "answer"])
    rubric = vf.Rubric(
        funcs=[rubric_obj.reward_func, rubric_obj.format_reward_func],
        weights=[1.0, 0.2],
        parser=parser,
    )

    vf_env = vf.SingleTurnEnv(
        dataset=to_hf(train_ds),
        eval_dataset=to_hf(val_ds),
        system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
        rubric=rubric,
    )
    return vf_env, rubric_obj


def run_verifiers_training(config: dict) -> None:
    """Launch training with verifiers.GRPOTrainer."""
    import verifiers as vf

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct")

    vf_env, _ = build_verifiers_env(config)
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    train_cfg = config.get("training", {})
    args = vf.grpo_defaults(
        run_name=f"bixbench-hypothesis-grpo-{int(time.time())}",
        num_train_epochs=train_cfg.get("n_epochs", 1),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        num_generations=config.get("grpo", {}).get("group_size", 8),
        max_new_tokens=config.get("grpo", {}).get("max_new_tokens", 2048),
        beta=float(config.get("grpo", {}).get("kl_coef", 0.01)),
    )

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=args,
    )
    trainer.train()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_mean_step_scores(trajectories: list[dict]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for t in trajectories:
        for k, v in t.get("step_scores", {}).items():
            sums[k] += v
            counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}


def _find_action_for_protocol_step(steps: list[dict], proto_step_name: str) -> str:
    """Return the action text most likely corresponding to a protocol step."""
    if not steps:
        return ""
    # Use last step's action as fallback; ideally use keyword matching
    return " ".join(str(s.get("action", "")) for s in steps)


def _find_logprob_for_protocol_step(
    steps: list[dict], proto_step_name: str
) -> float | None:
    """Return the log_prob for the most relevant conversation step."""
    for s in reversed(steps):
        lp = s.get("log_prob")
        if lp is not None:
            return lp
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-level GRPO training on Nemotron-RL-bixbench_hypothesis"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--use-verifiers-trainer",
        action="store_true",
        help="Use verifiers.GRPOTrainer instead of the async training loop",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.use_verifiers_trainer:
        run_verifiers_training(config)
    else:
        asyncio.run(train(config))


if __name__ == "__main__":
    main()
