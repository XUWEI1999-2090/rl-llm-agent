#!/bin/bash
# scripts/eval_bixbench.sh
# =========================
# 在 BixBench 上评估智能体性能
#
# 用法：
#   bash scripts/eval_bixbench.sh                    # 用 GPT-4o
#   MODEL=local bash scripts/eval_bixbench.sh        # 用本地 vLLM
#   bash scripts/eval_bixbench.sh --split test       # test split

set -euo pipefail

MODEL="${MODEL:-gpt-4o}"
MODEL_URL="${MODEL_URL:-}"        # 本地 vLLM: http://localhost:8000/v1
SPLIT="${1:-test}"
N_PARALLEL="${N_PARALLEL:-5}"
OUTPUT_DIR="results/eval_${MODEL}_$(date +%Y%m%d_%H%M%S)"

echo "=== BixBench 评估 ==="
echo "模型:       $MODEL"
echo "Split:      $SPLIT"
echo "并行 runs:  $N_PARALLEL per capsule"
echo "输出目录:   $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# 方式一：使用 BixBench 官方评估 harness（推荐）
# BixBench 仓库提供了 python -m bixbench.evaluate CLI
python -m bixbench.evaluate \
    --agent "src.agents.notebook_agent.make_bixbench_agent" \
    --agent-kwargs "{\"model_name\": \"$MODEL\", \"temperature\": 0.3}" \
    --env "src.envs.notebook_env.ControlledNotebookEnv" \
    --docker "futurehouse/bixbench-env:v1.0" \
    --dataset "hf://futurehouse/BixBench" \
    --split "$SPLIT" \
    --n-parallel "$N_PARALLEL" \
    --output "$OUTPUT_DIR/scores.json" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo ""
echo "=== 评估完成 ==="
echo "结果: $OUTPUT_DIR/scores.json"
cat "$OUTPUT_DIR/scores.json" | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'Open-answer accuracy:   {data.get(\"open_answer_accuracy\", 0):.3f}')
print(f'MCQ majority precision: {data.get(\"mcq_precision\", 0):.3f}')
print(f'Capsules evaluated:     {data.get(\"n_capsules\", 0)}')
"
