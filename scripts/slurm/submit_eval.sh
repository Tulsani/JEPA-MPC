#!/bin/bash
# Submit the C3/C4 closed-loop matrix (run from the repo root):
#   LEARNED=$RUNS_ROOT/pusht_slow_learned_b0.5_s0/best.pt \
#   FIXED=$RUNS_ROOT/pusht_slow_fixed_k3_s0/best.pt \
#   RANDOM_CKPT=$RUNS_ROOT/pusht_slow_random_u_s0/best.pt \
#   bash scripts/slurm/submit_eval.sh
# (Not RANDOM: that is a bash builtin that expands to a random number.)
# Optional: SEEDS="42 43 44" OFFSETS="50 75" BASELINES=0 (skip LeWM + flat) ABLATIONS=1
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
: "${LEARNED:?}" "${FIXED:?}" "${RANDOM_CKPT:?}"
OFFSETS=${OFFSETS:-"25 50 75"}
SEEDS=${SEEDS:-"42"}
BASELINES=${BASELINES:-1}
ABLATIONS=${ABLATIONS:-0}

submit() { echo "  $* -> $(env "$@" sbatch --parsable scripts/slurm/eval_planning.sh)"; }

for d in ${OFFSETS}; do
    for seed in ${SEEDS}; do
        echo "d=${d} seed=${seed}"
        if [ "${BASELINES}" = "1" ]; then
            submit PLANNER=lewm D=${d} SEED=${seed}
            submit CKPT=${LEARNED} MODE=flat D=${d} SEED=${seed}
        fi
        submit CKPT=${LEARNED}     MODE=learned  D=${d} SEED=${seed}
        submit CKPT=${FIXED}       MODE=fixed    D=${d} SEED=${seed}
        submit CKPT=${RANDOM_CKPT} MODE=duration D=${d} SEED=${seed}
        if [ "${ABLATIONS}" = "1" ]; then
            # Same learned checkpoint, different timing rule / macro prior.
            submit CKPT=${LEARNED} MODE=fixed FIXED_LENGTH=1 D=${d} SEED=${seed}
            submit CKPT=${LEARNED} MODE=fixed FIXED_LENGTH=3 D=${d} SEED=${seed}
            submit CKPT=${LEARNED} MODE=learned STD_MACROS=1 D=${d} SEED=${seed}
        fi
    done
done
