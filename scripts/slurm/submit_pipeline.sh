#!/bin/bash
# Submit the full day-1/2 pipeline with job dependencies (run from the repo root):
#   bash scripts/slurm/submit_pipeline.sh              # cache -> fast -> slow sweep
#   SKIP_CACHE=1 bash scripts/slurm/submit_pipeline.sh # cache already built
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

BETAS=${BETAS:-"0.25 0.5 1.0 2.0"}
KS=${KS:-"3 5 8"}

dependency=()
if [ -z "${SKIP_CACHE:-}" ]; then
    cache_id=$(sbatch --parsable scripts/slurm/cache_latents.sh)
    echo "cache:  ${cache_id}"
    dependency=(--dependency=afterok:${cache_id})
fi

fast_id=$(sbatch --parsable "${dependency[@]}" scripts/slurm/train_fast.sh)
echo "fast:   ${fast_id}"

for beta in ${BETAS}; do
    id=$(MODE=learned BETA=${beta} sbatch --parsable --dependency=afterok:${fast_id} scripts/slurm/train_slow.sh)
    echo "learned beta=${beta}: ${id}"
done
for k in ${KS}; do
    id=$(MODE=fixed K=${k} sbatch --parsable --dependency=afterok:${fast_id} scripts/slurm/train_slow.sh)
    echo "fixed K=${k}: ${id}"
done
id=$(MODE=random sbatch --parsable --dependency=afterok:${fast_id} scripts/slurm/train_slow.sh)
echo "random: ${id}"
