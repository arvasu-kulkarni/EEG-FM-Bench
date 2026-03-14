#!/usr/bin/env bash
# ============================================================================
# run_ndx_sweep.sh — Sweep all pretrained epoch checkpoints × {FT, LP}
#
# Usage:
#   ./scripts/run_ndx_sweep.sh <pretrained_dir> <external_model_py> [gpu_list] [epoch_list] [train_config_yaml]
#
# Examples:
#   ./scripts/run_ndx_sweep.sh /share/tmp/mishra/output/20260227_093832 \
#       /home/neurodx/adityaraymishra/ndx-pipeline/model.py
#
#   ./scripts/run_ndx_sweep.sh /share/tmp/mishra/output/20260228_011414 \
#       /home/neurodx/adityaraymishra/ndx-pipeline/model.py 0,1,2,3,4,5,6,7
#
#   ./scripts/run_ndx_sweep.sh /share/tmp/mishra/output/2026-03-13_20-25-46 \
#       /home/neurodx/adityaraymishra/ndx-pipeline-neurodx/model.py \
#       0,1,2,3,4,5,6,7 "" /home/neurodx/adityaraymishra/ndx-pipeline-neurodx/configs/trainconfig.yaml
#
# Output structure:
#   runs/<timestamp>/
#     epoch_1/ft/   epoch_1/lp/
#     epoch_2/ft/   epoch_2/lp/
#     ...
#     best_results_ft.txt   best_results_lp.txt   best_results_combined.txt
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# --------------- Python interpreter ---------------
# Prefer the venv alongside this repo; fall back to whatever is on PATH.
VENV_PYTHON="$(dirname "$ROOT_DIR")/.venv/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON="$VENV_PYTHON"
else
    PYTHON="$(command -v python3 || command -v python)"
fi
# --------------- Arguments ---------------
PRETRAINED_DIR="${1:?Usage: $0 <pretrained_dir> <external_model_py> [gpu_list] [epoch_list]}"
EXTERNAL_MODEL_PY="${2:?Usage: $0 <pretrained_dir> <external_model_py> [gpu_list] [epoch_list]}"
GPU_LIST="${3:-0,1,2,3,4,5,6,7}"
EPOCH_FILTER="${4:-}"  # Optional: comma-separated epochs e.g. "12,13,14,15,16,17,18"
TRAIN_CONFIG_PATH="${5:-}"  # Optional: explicit ndx-pipeline-neurodx trainconfig.yaml

CONF_FILE="baseline/manas/manas_ndx_sweep_template.yaml"

# --------------- Derived paths ---------------
RUN_TIMESTAMP="$(basename "$PRETRAINED_DIR")"
SWEEP_ROOT="$ROOT_DIR/runs/$RUN_TIMESTAMP"
mkdir -p "$SWEEP_ROOT"

echo "======================================="
echo "NDX Sweep: $RUN_TIMESTAMP"
echo "  Pretrained dir : $PRETRAINED_DIR"
echo "  External model : $EXTERNAL_MODEL_PY"
echo "  GPUs           : $GPU_LIST"
echo "  Epoch filter   : ${EPOCH_FILTER:-all}"
echo "  Train config   : ${TRAIN_CONFIG_PATH:-auto}"
echo "  Sweep root     : $SWEEP_ROOT"
echo "======================================="

# --------------- Runtime directories ---------------
RUNTIME_DIR="$ROOT_DIR/.local_runtime"
CACHE_DIR="$ROOT_DIR/eegfmbench_cache"
mkdir -p "$RUNTIME_DIR" "$CACHE_DIR"
mkdir -p "$RUNTIME_DIR/wandb" "$RUNTIME_DIR/wandb_cache" "$RUNTIME_DIR/wandb_config"
mkdir -p "$RUNTIME_DIR/xdg_cache" "$RUNTIME_DIR/hf_home" "$RUNTIME_DIR/hf_datasets"
mkdir -p "$RUNTIME_DIR/torch_home" "$RUNTIME_DIR/mpl"

export EEGFM_PROJECT_ROOT="$ROOT_DIR"
export EEGFM_CONF_ROOT="$ROOT_DIR/assets/conf"
export DATABASE_PROC_ROOT="${DATABASE_PROC_ROOT:-/share/data/eegfmbench/proc}"
export DATABASE_RAW_ROOT="${DATABASE_RAW_ROOT:-/share/data/eegfmbench/raw}"
export DATABASE_CACHE_ROOT="$CACHE_DIR"

export WANDB_DIR="$RUNTIME_DIR/wandb"
export WANDB_CACHE_DIR="$RUNTIME_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$RUNTIME_DIR/wandb_config"

export XDG_CACHE_HOME="$RUNTIME_DIR/xdg_cache"
export HF_HOME="$RUNTIME_DIR/hf_home"
export HF_DATASETS_CACHE="$RUNTIME_DIR/hf_datasets"
export TORCH_HOME="$RUNTIME_DIR/torch_home"
export MPLCONFIGDIR="$RUNTIME_DIR/mpl"
export MNE_DONTWRITE_HOME="true"

# --------------- Discover epoch checkpoints (numerically sorted) ---------------
EPOCH_FILES=( $(ls "$PRETRAINED_DIR"/mae_epoch_*.pt 2>/dev/null | sort -V) )
if [[ ${#EPOCH_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No mae_epoch_*.pt files found in $PRETRAINED_DIR"
    exit 1
fi

EPOCHS=()
for f in "${EPOCH_FILES[@]}"; do
    ep=$(basename "$f" | sed -n 's/mae_epoch_\([0-9]*\)\.pt/\1/p')
    EPOCHS+=("$ep")
done

# Apply epoch filter if specified
if [[ -n "$EPOCH_FILTER" ]]; then
    IFS=',' read -ra FILTER_EPOCHS <<< "$EPOCH_FILTER"
    FILTERED=()
    for ep in "${EPOCHS[@]}"; do
        for fep in "${FILTER_EPOCHS[@]}"; do
            if [[ "$ep" == "$fep" ]]; then
                FILTERED+=("$ep")
                break
            fi
        done
    done
    EPOCHS=("${FILTERED[@]}")
fi

echo "Found ${#EPOCHS[@]} epoch checkpoints to process: ${EPOCHS[*]}"

# --------------- Build job list ---------------
# Each job = (epoch, method)
declare -a JOB_EPOCHS
declare -a JOB_METHODS
for ep in "${EPOCHS[@]}"; do
    for method in lp ft; do
        JOB_EPOCHS+=("$ep")
        JOB_METHODS+=("$method")
    done
done

TOTAL_JOBS=${#JOB_EPOCHS[@]}
echo "Total jobs: $TOTAL_JOBS (${#EPOCHS[@]} epochs × 2 methods)"

# --------------- GPU assignment ---------------
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

# --------------- Unique master_port allocator ---------------
BASE_PORT=51300

# --------------- Launch jobs in batches ---------------
JOB_IDX=0
BATCH_NUM=0

while [[ $JOB_IDX -lt $TOTAL_JOBS ]]; do
    BATCH_NUM=$((BATCH_NUM + 1))
    PIDS=()
    BATCH_SIZE=0
    GPU_IDX=0

    echo ""
    echo "--- Batch $BATCH_NUM ---"

    while [[ $GPU_IDX -lt $NUM_GPUS && $JOB_IDX -lt $TOTAL_JOBS ]]; do
        EP="${JOB_EPOCHS[$JOB_IDX]}"
        METHOD="${JOB_METHODS[$JOB_IDX]}"
        GPU="${GPUS[$GPU_IDX]}"

        if [[ "$METHOD" == "lp" ]]; then
            TRAIN_METHOD="linear_probe"
        else
            TRAIN_METHOD="full_ft"
        fi

        RUN_DIR="$SWEEP_ROOT/epoch_${EP}/${METHOD}"
        LOG_FILE="$RUN_DIR/sweep.log"
        EXPERIMENT_NAME="manas_ndx_${RUN_TIMESTAMP}_epoch${EP}_${METHOD}"
        MASTER_PORT=$((BASE_PORT + JOB_IDX))

        mkdir -p "$RUN_DIR"

        echo "  [GPU $GPU] epoch_${EP}/${METHOD} -> $RUN_DIR"

        EXTRA_ARGS=()
        if [[ -n "$TRAIN_CONFIG_PATH" ]]; then
            EXTRA_ARGS+=(model.ndx_train_config_path="$TRAIN_CONFIG_PATH")
        fi

        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" baseline_main.py \
            conf_file="$CONF_FILE" \
            model_type=manas \
            model.pretrained_path="$PRETRAINED_DIR" \
            model.pretrained_epoch="$EP" \
            model.mae_type=ndx \
            model.external_model_py="$EXTERNAL_MODEL_PY" \
            "${EXTRA_ARGS[@]}" \
            training.train_method="$TRAIN_METHOD" \
            logging.run_dir="$RUN_DIR" \
            logging.experiment_name="$EXPERIMENT_NAME" \
            sweep_mode=true \
            sweep_root="$SWEEP_ROOT" \
            master_port="$MASTER_PORT" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
        JOB_IDX=$((JOB_IDX + 1))
        GPU_IDX=$((GPU_IDX + 1))
        BATCH_SIZE=$((BATCH_SIZE + 1))
    done

    echo "  Launched $BATCH_SIZE jobs. Waiting for batch to finish..."

    # Wait for all jobs in this batch
    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: Job PID $pid exited with non-zero status"
            FAIL=$((FAIL + 1))
        fi
    done

    if [[ $FAIL -gt 0 ]]; then
        echo "  $FAIL / $BATCH_SIZE jobs failed in batch $BATCH_NUM (check individual sweep.log files)"
    else
        echo "  Batch $BATCH_NUM completed successfully."
    fi
done

echo ""
echo "========================================"
echo "Sweep complete for $RUN_TIMESTAMP"
echo "  Results: $SWEEP_ROOT/"
echo "  Model-level summary: $SWEEP_ROOT/best_results_*.txt"
echo "========================================"

# Run standalone aggregation as final pass (in case some live updates were missed)
if [[ -f "$ROOT_DIR/scripts/aggregate_sweep.py" ]]; then
    echo "Running final aggregation..."
    "$PYTHON" "$ROOT_DIR/scripts/aggregate_sweep.py" "$SWEEP_ROOT"
fi
