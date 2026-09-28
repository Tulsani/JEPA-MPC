#!/bin/bash
# One-time environment setup. Run on a login node (needs internet):
#   bash scripts/slurm/setup_env.sh
set -euo pipefail

SCRATCH=${SCRATCH_ROOT:-/gpfs/scratch/${USER}/at6646}
ENV_PATH=${ENV_PATH:-${SCRATCH}/conda_envs/jepa_mpc}

source ~/.bashrc
if [ ! -d "${ENV_PATH}" ]; then
    conda create -y -p "${ENV_PATH}" python=3.10
fi
conda activate "${ENV_PATH}"
# gymnasium[all] (pulled in by stable-worldmodel[env]) builds box2d-py, which needs swig.
conda install -y -c conda-forge swig ffmpeg

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[cluster,dev]"

python -m pytest -q
python -c "import stable_worldmodel as swm, h5py, hdf5plugin; print('stable-worldmodel', swm.__version__ if hasattr(swm, '__version__') else 'ok')"
echo "Environment ready at ${ENV_PATH}"
