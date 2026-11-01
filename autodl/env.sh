#!/usr/bin/env bash
# Source this file in every new shell: source autodl/env.sh
HQA_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HQA_PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$HQA_PROJECT_ROOT/.venv/bin/activate"
fi
cd "$HQA_PROJECT_ROOT"
export PYTHONPATH="$HQA_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MINARI_DATASETS_PATH="${MINARI_DATASETS_PATH:-$HQA_PROJECT_ROOT/minari_datasets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
