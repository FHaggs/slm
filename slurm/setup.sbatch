#!/bin/bash
# Instala o ambiente no no de computacao. Requer uv: python3 -m pip install --user uv
#SBATCH -J slm-setup
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 30
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"
export UV_CACHE_DIR="$SLM_SCRATCH/uv-cache"
export UV_LINK_MODE=copy

mkdir -p "$SLM_SCRATCH"/{data,checkpoints}
uv python install 3.12
uv sync
uv run python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"
