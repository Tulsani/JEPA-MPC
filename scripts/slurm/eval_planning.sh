#!/bin/bash
#SBATCH --job-name=jepa_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00

# Closed-loop Push-T evaluation (C3/C4).
#   PLANNER=lewm D=25 sbatch scripts/slurm/eval_planning.sh                 # reproduce flat LeWM
#   CKPT=$RUNS_ROOT/pusht_slow_learned_b0.5_s0/best.pt MODE=learned D=75 sbatch scripts/slurm/eval_planning.sh
#   CKPT=... MODE=flat D=75 sbatch scripts/slurm/eval_planning.sh           # our flat baseline
# BUDGET defaults to 2*D env steps.

source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"

PLANNER=${PLANNER:-two_clock}
D=${D:-25}
BUDGET=${BUDGET:-$((2 * D))}
SEED=${SEED:-42}
NUM_EVAL=${NUM_EVAL:-50}
EXTRA=()
if [ "${PLANNER}" = "two_clock" ]; then
    : "${CKPT:?set CKPT to a slow-stage checkpoint}"
    EXTRA+=(--checkpoint "${CKPT}")
    if [ -n "${MODE:-}" ]; then EXTRA+=(--mode "${MODE}"); fi
    if [ -n "${COST_MODE:-}" ]; then EXTRA+=(--cost-mode "${COST_MODE}"); fi
    if [ -n "${VAL_ONLY:-}" ]; then EXTRA+=(--val-only); fi
    RUN_TAG=$(basename "$(dirname "${CKPT}")")_${MODE:-ckpt}
else
    RUN_TAG=lewm_flat
fi
OUT=${OUT:-${RUNS_ROOT}/eval/${RUN_TAG}_d${D}_seed${SEED}}

python scripts/eval_planning.py --planner "${PLANNER}" \
    --goal-offset "${D}" --eval-budget "${BUDGET}" --num-eval "${NUM_EVAL}" --seed "${SEED}" \
    --dataset-dir "${DATA_ROOT}/raw" --out "${OUT}" ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done: $(date)"
