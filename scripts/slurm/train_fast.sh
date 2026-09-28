#!/bin/bash
#SBATCH --job-name=jepa_fast
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00

# Stage 1: fast clock (GRU latent dynamics) on cached LeWM latents.
#   sbatch scripts/slurm/train_fast.sh
#   SEED=1 EPOCHS=40 sbatch scripts/slurm/train_fast.sh

source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"

CONFIG=${CONFIG:-configs/pusht_two_clock.yaml}
SEED=${SEED:-0}
EXP_NAME=${EXP_NAME:-pusht_fast_s${SEED}}
EXTRA=()
if [ -n "${EPOCHS:-}" ]; then EXTRA+=(--epochs "${EPOCHS}"); fi

python scripts/train_two_clock.py --config "${CONFIG}" --stage fast --seed "${SEED}" \
    --out "${RUNS_ROOT}/${EXP_NAME}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done: $(date)"
