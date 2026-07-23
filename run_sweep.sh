#!/usr/bin/env bash
# =============================================================================
#  run_sweep.sh  —  Launch 8 parallel training runs, one per MI300X GPU
#  Project: multi_agent_nav_max
#
#  Usage:
#    bash run_sweep.sh               # uses all 8 GPUs
#    bash run_sweep.sh 0 1 2 3       # use only GPUs 0-3
#
#  Each run gets its own:
#    logs_max_run<N>/metrics.csv
#    checkpoints_max_run<N>/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Activate env + ROCm vars
CONDA_BASE=$(conda info --base 2>/dev/null)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate marl
[[ -f "${HOME}/.bashrc_rocm" ]] && source "${HOME}/.bashrc_rocm"

# GPU list: use args if provided, else all 8
if [[ $# -gt 0 ]]; then
    GPU_LIST=("$@")
else
    GPU_LIST=(0 1 2 3 4 5 6 7)
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   multi_agent_nav_max — Parallel Sweep (${#GPU_LIST[@]} GPU(s))     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PIDS=()
LOG_FILES=()

for GPU_IDX in "${GPU_LIST[@]}"; do
    RUN_ID="run${GPU_IDX}"
    LOG_DIR="${SCRIPT_DIR}/logs_max_${RUN_ID}"
    CKPT_DIR="${SCRIPT_DIR}/checkpoints_max_${RUN_ID}"
    LOG_FILE="${LOG_DIR}/train.log"

    mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

    echo "  [GPU ${GPU_IDX}] Launching → ${LOG_FILE}"

    # Each process sees only ONE GPU so JAX uses device index 0 internally.
    # The script's checkpoint/log paths are overridden via env vars below.
    ROCR_VISIBLE_DEVICES="${GPU_IDX}" \
    HIP_VISIBLE_DEVICES="${GPU_IDX}" \
    JAX_PLATFORMS=rocm \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    MARL_LOG_DIR="${LOG_DIR}" \
    MARL_CKPT_DIR="${CKPT_DIR}" \
    python train_pure_jax.py \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    LOG_FILES+=("${LOG_FILE}")
done

echo ""
echo "  Launched ${#PIDS[@]} job(s)."
echo "  PIDs: ${PIDS[*]}"
echo ""
echo "  Monitor live:"
echo "    tail -f logs_max_run0/train.log"
echo ""
echo "  Kill all:"
echo "    kill ${PIDS[*]}"
echo ""

# Wait for all and report
ALL_OK=true
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    LOG="${LOG_FILES[$i]}"
    if wait "${PID}"; then
        echo "  [✓] PID ${PID} finished OK  →  ${LOG}"
    else
        echo "  [✗] PID ${PID} exited with error  →  check ${LOG}"
        ALL_OK=false
    fi
done

echo ""
if ${ALL_OK}; then
    echo "✅  All runs completed successfully."
else
    echo "⚠️   Some runs had errors — check individual log files."
    exit 1
fi
