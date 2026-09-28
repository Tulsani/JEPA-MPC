#!/bin/bash
#SBATCH --job-name=netime_foundation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-14:00:00

# ── Usage ─────────────────────────────────────────────────────────────────
# One-time setup (login node, needs internet — Weather/Exchange/ECL/Traffic
# are only reachable via huggingface_hub, not from most compute nodes):
#   bash scripts/download_foundation_data.sh
#
# Default run (pretrain ETTh1+ETTh2+ETTm1, zero-shot ETTm2/Weather/Exchange/ECL/Traffic):
#   sbatch scripts/train_foundation.sh
#
# Override corpus / model size / hyperparams:
#   MODEL_SIZE=large EPOCHS=80 sbatch scripts/train_foundation.sh
#   PRETRAIN_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2" ZERO_SHOT_DATASETS="Weather Exchange ECL Traffic" \
#       sbatch scripts/train_foundation.sh
#
# Euclidean-ablation control (identical architecture/corpus/hyperparams, Poincare-ball
# mapping replaced by identity — isolates the effect of hyperbolic geometry itself):
#   GEOMETRY=euclidean EXP_NAME=nano_wecm1_sw_lr3e4_euclidean \
#       MODEL_SIZE=nano SIZE_WEIGHTED_SAMPLING=true LR=3e-4 \
#       PRETRAIN_DATASETS="Weather Exchange ECL ETTm1" ZERO_SHOT_DATASETS="ETTh1 ETTh2 ETTm2 Traffic" \
#       sbatch scripts/train_foundation.sh
#
# See train_foundation.py for full protocol details (joint CI/univariate
# pretraining on one shared HyperTimeV2, then zero-shot eval with no
# fine-tuning on held-out datasets).

set -e

USERNAME=${USER}
SCRATCH=/gpfs/scratch/${USERNAME}/at6646
ENV_PATH=${SCRATCH}/conda_envs/hypertime
PROJECT_DIR=${SLURM_SUBMIT_DIR}

# ── Config ────────────────────────────────────────────────────────────────
PRETRAIN_DATASETS=${PRETRAIN_DATASETS:-"ETTh1 ETTh2 ETTm1"}
ZERO_SHOT_DATASETS=${ZERO_SHOT_DATASETS:-"ETTm2 Weather Exchange ECL Traffic"}

MODEL_SIZE=${MODEL_SIZE:-medium}
SEQ_LEN=${SEQ_LEN:-336}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-3}
PATIENCE=${PATIENCE:-15}
SIZE_WEIGHTED_SAMPLING=${SIZE_WEIGHTED_SAMPLING:-false}
GEOMETRY=${GEOMETRY:-hyperbolic}
C_GLOBAL_INIT=${C_GLOBAL_INIT:-0.5}
C_MESO_INIT=${C_MESO_INIT:-1.0}
C_LOCAL_INIT=${C_LOCAL_INIT:-2.0}
ENC_OUT_INIT_STD=${ENC_OUT_INIT_STD:-0.01}
DECOMPOSER=${DECOMPOSER:-fixed_ma}

# Overfitting-fix knobs (see train.py / model_no_attn_upd.py for details)
GEO_DROPOUT=${GEO_DROPOUT:-0.2}
HYP_HIDDEN_SCALE=${HYP_HIDDEN_SCALE:-1.0}
CURVATURE_WD=${CURVATURE_WD:-1e-3}
PROJ_HIDDEN=${PROJ_HIDDEN:-64}

EXP_NAME=${EXP_NAME:-foundation_${MODEL_SIZE}}

# ── Environment ───────────────────────────────────────────────────────────
source ~/.bashrc
conda activate ${ENV_PATH}

export HF_HOME=${SCRATCH}/hf_cache
export TORCH_HOME=${SCRATCH}/torch_cache
export TMPDIR=${SCRATCH}/tmp
mkdir -p ${SCRATCH}/tmp

cd ${PROJECT_DIR}
mkdir -p logs outputs_foundation data

# ── Log job info ──────────────────────────────────────────────────────────
echo "================================================"
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Node:                ${SLURMD_NODENAME}"
echo "Project dir:         ${PROJECT_DIR}"
echo "Pretrain datasets:   ${PRETRAIN_DATASETS}"
echo "Zero-shot datasets:  ${ZERO_SHOT_DATASETS}"
echo "Model size:          ${MODEL_SIZE}"
echo "Seq len:             ${SEQ_LEN}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Epochs:              ${EPOCHS}"
echo "LR:                  ${LR}"
echo "Patience:            ${PATIENCE}"
echo "Size-weighted sampl: ${SIZE_WEIGHTED_SAMPLING}"
echo "Geometry:            ${GEOMETRY}"
echo "Curvature inits (g/m/l): ${C_GLOBAL_INIT} / ${C_MESO_INIT} / ${C_LOCAL_INIT}"
echo "Enc out_init_std:    ${ENC_OUT_INIT_STD}"
echo "Decomposer:          ${DECOMPOSER}"
echo "Exp name:            ${EXP_NAME}"
echo "--- overfitting fixes ---"
echo "geo_dropout:         ${GEO_DROPOUT}"
echo "hyp_hidden_scale:    ${HYP_HIDDEN_SCALE}"
echo "curvature_wd:        ${CURVATURE_WD}"
echo "proj_hidden:         ${PROJ_HIDDEN}"
echo "Start time:          $(date)"
echo "================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python -c "
import torch, sys
print(f'Python:     {sys.version.split()[0]}')
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:        {torch.cuda.get_device_name(0)}')
    print(f'VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# Fail fast with a clear message if the non-ETT datasets weren't pre-fetched.
for f in Weather Exchange ECL Traffic; do
    if [ ! -f "${PROJECT_DIR}/data/${f}.csv" ]; then
        echo "WARNING: ${PROJECT_DIR}/data/${f}.csv not found."
        echo "         Run 'bash scripts/download_foundation_data.sh' on a login node first"
        echo "         if this dataset is in PRETRAIN_DATASETS or ZERO_SHOT_DATASETS."
    fi
done
echo ""

# ── Run ───────────────────────────────────────────────────────────────────
echo "Starting foundation pretraining..."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA_ARGS=()
if [ "${SIZE_WEIGHTED_SAMPLING}" = "true" ]; then
    EXTRA_ARGS+=(--size_weighted_sampling)
fi

python train_foundation.py \
    --pretrain_datasets  ${PRETRAIN_DATASETS} \
    --zero_shot_datasets ${ZERO_SHOT_DATASETS} \
    --data_path          ${PROJECT_DIR}/data \
    --seq_len            ${SEQ_LEN} \
    --horizons           96 192 336 720 \
    --size               ${MODEL_SIZE} \
    --patch_size         16 \
    --patch_stride       8 \
    --train_stride       1 \
    --epochs             ${EPOCHS} \
    --warmup_epochs      ${WARMUP_EPOCHS} \
    --batch_size         ${BATCH_SIZE} \
    --lr                 ${LR} \
    --weight_decay       1e-4 \
    --patience           ${PATIENCE} \
    --num_workers        4 \
    --geo_dropout        ${GEO_DROPOUT} \
    --hyp_hidden_scale   ${HYP_HIDDEN_SCALE} \
    --curvature_wd       ${CURVATURE_WD} \
    --proj_hidden        ${PROJ_HIDDEN} \
    --geometry           ${GEOMETRY} \
    --c_global_init      ${C_GLOBAL_INIT} \
    --c_meso_init        ${C_MESO_INIT} \
    --c_local_init       ${C_LOCAL_INIT} \
    --enc_out_init_std   ${ENC_OUT_INIT_STD} \
    --decomposer         ${DECOMPOSER} \
    --exp_name           ${EXP_NAME} \
    --output_dir         ${PROJECT_DIR}/outputs_foundation \
    "${EXTRA_ARGS[@]}"

echo ""
echo "================================================"
echo "Done: $(date)"
echo "Outputs:"
ls -lh ${PROJECT_DIR}/outputs_foundation/ | grep "${EXP_NAME}"
echo "================================================"
