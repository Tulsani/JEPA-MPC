#!/bin/bash
#SBATCH --job-name=jepa_cache
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-12:00:00

# Download the LeWM Push-T dataset + checkpoint and cache frozen embeddings.
#   sbatch scripts/slurm/cache_latents.sh
#   DATASET=cube sbatch scripts/slurm/cache_latents.sh     # OGBench cube-single (stretch)
# Quick check on a few episodes first:
#   LIMIT_EPISODES=20 OUT_NAME=pusht_lewm_debug sbatch scripts/slurm/cache_latents.sh

source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/common.sh"

DATASET=${DATASET:-pusht}
LIMIT_EPISODES=${LIMIT_EPISODES:-}
case "${DATASET}" in
    pusht)
        REPO=quentinll/lewm-pusht; FILE=pusht_expert_train.h5.zst ;;
    cube)
        # Archive name on HF; check the extracted .h5 name in the log if this changes.
        REPO=quentinll/lewm-cube; FILE=cube_single_expert.tar.zst ;;
    *) echo "unknown DATASET=${DATASET}"; exit 1 ;;
esac
OUT_NAME=${OUT_NAME:-${DATASET}_lewm}

EXTRA=()
if [ -n "${LIMIT_EPISODES}" ]; then EXTRA+=(--limit-episodes "${LIMIT_EPISODES}"); fi

python scripts/cache_latents.py \
    --dataset-repo "${REPO}" --dataset-file "${FILE}" \
    --model-repo "${REPO}" \
    --data-root "${DATA_ROOT}/raw" \
    --out "${DATA_ROOT}/latents/${OUT_NAME}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "Done: $(date)"
ls -lh "${DATA_ROOT}/latents/${OUT_NAME}"
