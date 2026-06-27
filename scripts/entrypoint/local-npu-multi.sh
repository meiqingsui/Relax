#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Default environment configuration for local single-node development.
# This script handles process cleanup, environment setup, and Ray cluster startup.
# It is designed to be *sourced* by run-*.sh scripts when no external entrypoint
# (spmd-multinode.sh or ray-job-npu.sh) has been used.
#
# When an existing Ray cluster is detected (RAY_ADDRESS set and `ray status` OK),
# this script delegates to `ray-job-npu.sh` (source mode) instead of starting a new
# local Ray head node.
#
# Usage (from a run script):
#   source scripts/entrypoint/local-npu.sh
#
# Environment variables:
#   ASCEND_RT_VISIBLE_DEVICES   - Comma-separated NPU IDs (e.g., "0,1,2,3" → 4 NPUs)
#   MASTER_ADDR                 - Head node IP address (default: 127.0.0.1)
#   MEGATRON                    - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX                       - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job-npu.sh"
    return 0 2>/dev/null || exit 0
fi

set -eo pipefail

# ── process cleanup ─────────────────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
# pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
# pkill -9 python 2>/dev/null || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
unset MASTER_ADDR 2>/dev/null || true
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-/root/Megatron-Bridge/src/}
export MINDSPEED=${MINDSPEED:-/root/MindSpeed/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:${MEGATRON_BRIDGE_SRC}:${MINDSPEED}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── Ray cluster startup (single node) ──────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"80.5.25.115"}
export SOCKET_IFNAME="enp48s3u1u1"
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
NUM_GPUS="${NUM_GPUS:-16}"
NNODES="${WORLD_SIZE:-2}"

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    ray start --head \
        --node-ip-address ${MASTER_ADDR} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265

    sleep 5

    while true; do
        ray_status_output=$(ray status)
        gpu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        echo "Current GPU count: $gpu_count"
        gpu_count_int=$(echo "$gpu_count" | awk '{print int($1)}')
        device_count=$((gpu_count_int / ${NUM_GPUS}))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $gpu_count GPU resources)."
            ray status
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done

    # ── set entrypoint mode ────────────────────────────────────────────────────
    export RELAX_ENTRYPOINT_MODE="local"

    # Runtime env for single-node (empty, env inherited from Ray cluster)
    export RUNTIME_ENV_JSON="{
    \"env_vars\": {
        \"PYTHONUNBUFFERED\": \"1\",
        \"PYTHONPATH\": \"${PYTHONPATH}\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
        \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
        \"PYTORCH_NPU_ALLOC_CONF\": \"expandable_segments:True\"
    }
    }"

    echo "=== Local environment ready ==="
else
    # ── WORKER NODE ─────────────────────────────────────────────────────────
    echo "=== Worker node: joining Ray cluster at ${MASTER_ADDR}:6379 ==="
    while true; do
        ray start \
            --address="${MASTER_ADDR}:6379" \
            --node-ip-address "${CURRENT_IP}" \
            --disable-usage-stats \
            --dashboard-host=0.0.0.0 \
            --dashboard-port=8265

        sleep 5
        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
        fi
    done

    # Worker nodes block indefinitely (training runs on head node)
    echo "=== Worker node ready, waiting for training to complete ==="
    sleep inf
fi
