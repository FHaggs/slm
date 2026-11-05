#!/bin/bash
# Testes automatizados em CPU, sem ocupar GPU.
#SBATCH -J slm-tests
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 20
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${SLM_SCRATCH:-/scratch/$USER/slm}/uv-cache"
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=""
uv sync --quiet
uv run pytest -q "$@"
