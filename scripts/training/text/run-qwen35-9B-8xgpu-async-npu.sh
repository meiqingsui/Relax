#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xNPU (1-node) fully async training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/text/run-qwen35-9B-8xgpu-async-npu.sh

set -ex
set -o pipefail
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1536
export HCCL_OP_EXPANSION_MODE=AIV
export PYTHONPATH=/workspace/Megatron-Bridge/src/:/workspace/Megatron-LM/:/workspace/MindSpeed/:$PYTHONPATH
now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"
HOST_IP=$(hostname -I | awk '{print $1}')
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export HCCL_NPU_SOCKET_PORT_RANGE="62000-62050"
export HCCL_HOST_SOCKET_PORT_RANGE="62100-62200"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export TQ_CONTROLLER_GET_METADATA_TIMEOUT=20
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
# source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3.5-9B
   --ref-load ${EXP_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge

   #--load ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen35-9B_mcore_8xgpu_617/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${EXP_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 200
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --no-gradient-accumulation-fusion
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   #--use-dynamic-batch-size
   --qkv-format bshd
   --max-tokens-per-gpu 10240
   --micro-batch-size 2 # avoid OOM

   --no-rope-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-max-running-requests 256
   --sglang-cuda-graph-bs 1 2 4 8 16 32 64 128 192 256
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-attention-backend ascend
   --sglang-chunked-prefill-size 4096
   --sglang-max-prefill-tokens 8192
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-9B-8x-async-${now}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   --use-flash-attn
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}'\
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 32 \
   --ref-actor-config '{"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1, "max_tokens_per_gpu": 10240, "sequence_parallel": false, "only_load_weight": true}' \
   --fully-async \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9B-GRPO-gpu16-async-${now}.log
