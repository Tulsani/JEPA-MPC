#!/bin/bash
# Shared environment for UltraViolet jobs. Sourced by every sbatch script.
# Override any path by exporting it before `sbatch`.

set -euo pipefail

SCRATCH=${SCRATCH_ROOT:-/gpfs/scratch/${USER}/at6646}
ENV_PATH=${ENV_PATH:-${SCRATCH}/conda_envs/jepa_mpc}
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}

export DATA_ROOT=${DATA_ROOT:-${SCRATCH}/jepa_mpc_data}
export RUNS_ROOT=${RUNS_ROOT:-${SCRATCH}/jepa_mpc_runs}
export HF_HOME=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export STABLEWM_HOME=${SCRATCH}/stablewm
export TMPDIR=${SCRATCH}/tmp
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${TMPDIR}" "${DATA_ROOT}" "${RUNS_ROOT}" "${STABLEWM_HOME}"

# System bashrc files and conda hooks reference unset variables; relax -u here.
set +u
source ~/.bashrc || true
eval "$(conda shell.bash hook)"
conda activate "${ENV_PATH}"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs

echo "================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Project dir: ${PROJECT_DIR}"
echo "Git commit:  $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "DATA_ROOT:   ${DATA_ROOT}"
echo "RUNS_ROOT:   ${RUNS_ROOT}"
echo "Start time:  $(date)"
echo "================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "
import torch, sys
print(f'Python:  {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    torch.zeros(1, device='cuda')  # fails fast on a busy/broken GPU
" || {
    echo "ERROR: GPU unusable on ${SLURMD_NODENAME:-this node}. Resubmit with --exclude=${SLURMD_NODENAME:-<node>}" >&2
    exit 1
}
