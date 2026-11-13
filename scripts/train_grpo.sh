#!/usr/bin/env bash
# GRPO with Deletion Tuning, one run per call, with the paper's settings for each model.
#
# Usage: [METHOD=fft] bash scripts/train_grpo.sh <model> <output_dir> [extra Hydra overrides ...]
#   model: llama1b | llama3b | llama8b (GSM8K) or qwen0.5b | qwen3b | qwen7b | qwen14b | qwen30b_a3b (SimpleRL-Zoo)
# qwen14b runs on 2 nodes and qwen30b_a3b on 8 nodes of 8 GPUs, all in one Ray cluster.
# Masks are saved to <output_dir>/global_step_N/mask/ at every validation step, and <output_dir>/metrics.jsonl
# logs every step (src/summarize_grpo.py prints the best validation).
# METHOD=fft trains the full fine-tuning baseline instead and saves the model at the same steps to
# <output_dir>/global_step_N/actor/huggingface/.
# Run from the repository root after scripts/prepare_grpo_data.sh.
set -euo pipefail
MODEL=$1; OUT=$2; shift 2

ABEL=data/simplerl_zoo/simplelr_abel_level1to4/train.parquet
MOE_WRAP="[Qwen3MoeDecoderLayer,Qwen3MoeSparseMoeBlock]"
ABEL_EVAL=$(printf "data/simplerl_zoo/eval/abel/%s.parquet," gsm8k math500 minerva olympiadbench aime24 amc23)
case "$MODEL" in
  llama1b)
    ARGS=(--config-name gsm8k trainer.n_gpus_per_node=1
          actor_rollout_ref.model.path=meta-llama/Llama-3.2-1B-Instruct) ;;
  llama3b)
    ARGS=(--config-name gsm8k trainer.n_gpus_per_node=2
          actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct) ;;
  llama8b)
    ARGS=(--config-name gsm8k trainer.n_gpus_per_node=2
          actor_rollout_ref.model.path=meta-llama/Meta-Llama-3-8B-Instruct
          actor_rollout_ref.rollout.gpu_memory_utilization=0.5
          actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096) ;;
  qwen0.5b)
    ARGS=(--config-name simplerl_zoo actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B trainer.n_gpus_per_node=1
          +data.skip_chat_template=true
          "data.train_files=[$ABEL]"
          "data.val_files=[${ABEL_EVAL%,}]"
          "actor_rollout_ref.model.mask_calib_parquet_files=[$ABEL]") ;;
  qwen3b)
    ARGS=(--config-name simplerl_zoo actor_rollout_ref.model.path=Qwen/Qwen2.5-3B trainer.n_gpus_per_node=2
          actor_rollout_ref.rollout.gpu_memory_utilization=0.5) ;;
  qwen7b)
    ARGS=(--config-name simplerl_zoo actor_rollout_ref.model.path=Qwen/Qwen2.5-7B trainer.n_gpus_per_node=2
          actor_rollout_ref.rollout.gpu_memory_utilization=0.5
          actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096) ;;
  qwen14b)
    ARGS=(--config-name simplerl_zoo actor_rollout_ref.model.path=Qwen/Qwen2.5-14B
          trainer.n_gpus_per_node=8 trainer.nnodes=2
          actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
          actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
          actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
          actor_rollout_ref.rollout.tensor_model_parallel_size=1
          actor_rollout_ref.rollout.gpu_memory_utilization=0.5
          actor_rollout_ref.rollout.max_num_seqs=128
          actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
          data.val_batch_size=512) ;;
  qwen30b_a3b)
    ARGS=(--config-name simplerl_zoo actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Base
          trainer.n_gpus_per_node=8 trainer.nnodes=8
          actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
          actor_rollout_ref.actor.entropy_checkpointing=true
          actor_rollout_ref.rollout.tensor_model_parallel_size=4
          actor_rollout_ref.rollout.gpu_memory_utilization=0.45
          actor_rollout_ref.rollout.max_model_len=5120
          actor_rollout_ref.rollout.max_num_seqs=128
          actor_rollout_ref.rollout.enable_chunked_prefill=true
          actor_rollout_ref.rollout.enable_prefix_caching=true
          "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]"
          actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048
          actor_rollout_ref.nccl_timeout=3600
          data.val_batch_size=512
          # FSDP1, with each MoE block as its own unit.
          actor_rollout_ref.actor.strategy=fsdp
          "+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=$MOE_WRAP"
         ) ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac

case "${METHOD:-deletion_tuning}" in
  deletion_tuning) ;;
  fft)
    ARGS+=(actor_rollout_ref.model.deletion_tuning=false actor_rollout_ref.rollout.load_format=safetensors
           "actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]")
    case "$MODEL" in
      llama*) ARGS+=(actor_rollout_ref.actor.optim.lr=1e-6) ;;
      qwen*) ARGS+=(actor_rollout_ref.actor.optim.lr=5e-7) ;;
    esac
    if [ "$MODEL" = llama8b ]; then
      ARGS+=(actor_rollout_ref.actor.fsdp_config.param_offload=true
             actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
             actor_rollout_ref.actor.strategy=fsdp)
    fi ;;
  *) echo "METHOD must be deletion_tuning or fft" >&2; exit 1 ;;
esac

# Optional env override for the HF model path (e.g. ungated mirrors when no HF token).
if [ -n "${BASE_MODEL_OVERRIDE:-}" ]; then
  ARGS=("${ARGS[@]/actor_rollout_ref.model.path=*/actor_rollout_ref.model.path=$BASE_MODEL_OVERRIDE}")
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
export VERL_FILE_LOGGER_PATH="$OUT/metrics.jsonl"
python -m verl.trainer.main_ppo "${ARGS[@]}" \
    trainer.experiment_name="$(basename "$OUT")" \
    trainer.default_local_dir="$OUT" \
    "$@"
