#!/bin/bash
#SBATCH --job-name=jepa_slow
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00

# Stage 2: boundary gate + slow clock on a frozen fast clock.
#   MODE=learned BETA=0.5 sbatch scripts/slurm/train_slow.sh
#   MODE=fixed K=5        sbatch scripts/slurm/train_slow.sh
#   MODE=random           sbatch scripts/slurm/train_slow.sh

source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"

CONFIG=${CONFIG:-configs/pusht_two_clock.yaml}
SEED=${SEED:-0}
MODE=${MODE:-learned}
BETA=${BETA:-0.5}
K=${K:-5}
FAST_CKPT=${FAST_CKPT:-${RUNS_ROOT}/pusht_fast_s0/best.pt}
case "${MODE}" in
    learned) TAG=b${BETA} ;;
    fixed)   TAG=k${K} ;;
    random)  TAG=u ;;
    *) echo "unknown MODE=${MODE}"; exit 1 ;;
esac
EXP_NAME=${EXP_NAME:-pusht_slow_${MODE}_${TAG}_s${SEED}}
EXTRA=()
if [ -n "${EPOCHS:-}" ]; then EXTRA+=(--epochs "${EPOCHS}"); fi

python scripts/train_two_clock.py --config "${CONFIG}" --stage slow --seed "${SEED}" \
    --fast-checkpoint "${FAST_CKPT}" --segment-mode "${MODE}" \
    --boundary-cost "${BETA}" --fixed-length "${K}" \
    --out "${RUNS_ROOT}/${EXP_NAME}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done: $(date)"
