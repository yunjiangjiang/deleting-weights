#!/usr/bin/env bash
# Evaluate a GRPO Deletion Tuning mask, a LoRA adapter (global_step_N/actor/lora_adapter/), a full fine-tuned model
# (global_step_N/actor/huggingface/), or the base model (with "base") on its model's validation benchmarks:
# GSM8K for Llama, the six SimpleRL-Zoo benchmarks for Qwen. Prints the accuracy per benchmark and their mean.
#
# Usage: bash scripts/evaluate_grpo.sh <model> <mask_dir|lora_dir|model_dir|base> <output_dir>
#   model: llama1b | llama3b | llama8b | qwen0.5b | qwen3b | qwen7b | qwen14b | qwen30b_a3b
# Run from the repository root after scripts/prepare_grpo_data.sh.
set -euo pipefail
MODEL=$1; CHECKPOINT=$2; OUT=$3

GSM8K=(data/gsm8k/test.parquet)
SIMPLERL_QWEN=(data/simplerl_zoo/eval/qwen/{gsm8k,math500,minerva,olympiadbench,aime24,amc23}.parquet)
SIMPLERL_ABEL=(data/simplerl_zoo/eval/abel/{gsm8k,math500,minerva,olympiadbench,aime24,amc23}.parquet)
ARGS=()
case "$MODEL" in
  llama1b) BASE=meta-llama/Llama-3.2-1B-Instruct; VAL=("${GSM8K[@]}") ;;
  llama3b) BASE=meta-llama/Llama-3.2-3B-Instruct; VAL=("${GSM8K[@]}") ;;
  llama8b) BASE=meta-llama/Meta-Llama-3-8B-Instruct; VAL=("${GSM8K[@]}") ;;
  qwen0.5b) BASE=Qwen/Qwen2.5-0.5B; VAL=("${SIMPLERL_ABEL[@]}"); ARGS=(--skip_chat_template) ;;
  qwen3b) BASE=Qwen/Qwen2.5-3B; VAL=("${SIMPLERL_QWEN[@]}") ;;
  qwen7b) BASE=Qwen/Qwen2.5-7B; VAL=("${SIMPLERL_QWEN[@]}") ;;
  qwen14b) BASE=Qwen/Qwen2.5-14B; VAL=("${SIMPLERL_QWEN[@]}") ;;
  qwen30b_a3b) BASE=Qwen/Qwen3-30B-A3B-Base; VAL=("${SIMPLERL_QWEN[@]}") ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac

# Optional env override for the HF model path (e.g. ungated mirrors when no HF token).
BASE=${BASE_MODEL_OVERRIDE:-$BASE}

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python src/evaluate_grpo.py --model_name "$BASE" --checkpoint "$CHECKPOINT" --val_parquets "${VAL[@]}" \
  --output_dir "$OUT" "${ARGS[@]}"
