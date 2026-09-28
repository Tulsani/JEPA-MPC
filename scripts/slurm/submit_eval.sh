#!/bin/bash
# Submit the C3/C4 closed-loop matrix (run from the repo root):
#   LEARNED=$RUNS_ROOT/pusht_slow_learned_b0.5_s0/best.pt \
#   FIXED=$RUNS_ROOT/pusht_slow_fixed_k5_s0/best.pt \
#   RANDOM=$RUNS_ROOT/pusht_slow_random_u_s0/best.pt \
#   bash scripts/slurm/submit_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
: "${LEARNED:?}" "${FIXED:?}" "${RANDOM:?}"
OFFSETS=${OFFSETS:-"25 50 75"}
SEEDS=${SEEDS:-"42"}

for d in ${OFFSETS}; do
    for seed in ${SEEDS}; do
        echo "d=${d} seed=${seed}"
        PLANNER=lewm D=${d} SEED=${seed} sbatch --parsable scripts/slurm/eval_planning.sh
        CKPT=${LEARNED} MODE=flat    D=${d} SEED=${seed} sbatch --parsable scripts/slurm/eval_planning.sh
        CKPT=${LEARNED} MODE=learned D=${d} SEED=${seed} sbatch --parsable scripts/slurm/eval_planning.sh
        CKPT=${FIXED}   MODE=fixed   D=${d} SEED=${seed} sbatch --parsable scripts/slurm/eval_planning.sh
        CKPT=${RANDOM}  MODE=duration D=${d} SEED=${seed} sbatch --parsable scripts/slurm/eval_planning.sh
    done
done
