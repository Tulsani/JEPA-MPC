#!/bin/bash
#SBATCH --job-name=jepa_analyze
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00

# Offline C1/C2 analysis over every finished slow run for a seed.
#   sbatch scripts/slurm/analyze.sh
#   SEED=1 OUT_NAME=analysis_s1 sbatch scripts/slurm/analyze.sh
#   MAX_EPISODES=500 sbatch scripts/slurm/analyze.sh       # faster, subset of val episodes

source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"

SEED=${SEED:-0}
OUT_NAME=${OUT_NAME:-analysis_s${SEED}}
ARGS=()
for run in "${RUNS_ROOT}"/pusht_slow_*_s${SEED}; do
    if [ -f "${run}/best.pt" ]; then
        label=$(basename "${run}" | sed -e "s/^pusht_slow_//" -e "s/_s${SEED}$//")
        ARGS+=(--checkpoint "${label}=${run}/best.pt")
    fi
done
if [ ${#ARGS[@]} -eq 0 ]; then echo "no finished slow runs for seed ${SEED}"; exit 1; fi

if [ -n "${MAX_EPISODES:-}" ]; then ARGS+=(--max-episodes "${MAX_EPISODES}"); fi

python scripts/analyze_offline.py ${ARGS[@]+"${ARGS[@]}"} --out "${RUNS_ROOT}/${OUT_NAME}"
python scripts/collect_results.py --runs-root "${RUNS_ROOT}"
echo "Done: $(date)"
