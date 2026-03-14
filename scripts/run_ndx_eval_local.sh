#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONF_FILE="${1:-baseline/manas/manas_ndx_20260225_221743.yaml}"
MODEL_TYPE="${2:-manas}"
RUN_SUBDIR="${3:-}"

# Keep all runtime writes local to EEG-FM-Bench.
RUNTIME_DIR="$ROOT_DIR/.local_runtime"
if [[ -n "$RUN_SUBDIR" ]]; then
  if [[ "$RUN_SUBDIR" = /* ]]; then
    RUN_DIR="$RUN_SUBDIR"
  else
    RUN_DIR="$ROOT_DIR/eegfmbench_run/$RUN_SUBDIR"
  fi
else
  RUN_DIR="${EEGFMBENCH_RUN_DIR:-$ROOT_DIR/eegfmbench_run}"
fi
CACHE_DIR="$ROOT_DIR/eegfmbench_cache"
mkdir -p "$RUNTIME_DIR" "$RUN_DIR" "$CACHE_DIR"
mkdir -p "$RUNTIME_DIR/wandb" "$RUNTIME_DIR/wandb_cache" "$RUNTIME_DIR/wandb_config"
mkdir -p "$RUNTIME_DIR/xdg_cache" "$RUNTIME_DIR/hf_home" "$RUNTIME_DIR/hf_datasets"
mkdir -p "$RUNTIME_DIR/torch_home" "$RUNTIME_DIR/mpl"

export EEGFM_PROJECT_ROOT="$ROOT_DIR"
export EEGFM_CONF_ROOT="$ROOT_DIR/assets/conf"
export DATABASE_PROC_ROOT="${DATABASE_PROC_ROOT:-/share/data/eegfmbench/proc}"
export DATABASE_RAW_ROOT="${DATABASE_RAW_ROOT:-/share/data/eegfmbench/raw}"
export DATABASE_CACHE_ROOT="$CACHE_DIR"

export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="$RUNTIME_DIR/wandb"
export WANDB_CACHE_DIR="$RUNTIME_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$RUNTIME_DIR/wandb_config"

export XDG_CACHE_HOME="$RUNTIME_DIR/xdg_cache"
export HF_HOME="$RUNTIME_DIR/hf_home"
export HF_DATASETS_CACHE="$RUNTIME_DIR/hf_datasets"
export TORCH_HOME="$RUNTIME_DIR/torch_home"
export MPLCONFIGDIR="$RUNTIME_DIR/mpl"
export MNE_DONTWRITE_HOME="true"

python baseline_main.py \
  conf_file="$CONF_FILE" \
  model_type="$MODEL_TYPE" \
  logging.run_dir="$RUN_DIR" \
  logging.use_cloud=false \
  logging.cloud_backend=none
