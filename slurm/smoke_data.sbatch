#!/bin/bash
# Fase 1: dados reais em pequena quantidade para o pipeline em miniatura. Nao usa GPU.
#SBATCH -J slm-smoke-data
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 60
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
source .venv/bin/activate
export SLM_SCRATCH="${SLM_SCRATCH:-/scratch/$USER/slm}"
export HF_HOME="$SLM_SCRATCH/hf-cache"
export CUDA_VISIBLE_DEVICES=""
data="$SLM_SCRATCH/data/smoke"

python scripts/tokenize_pretrain.py --out "$data/pretrain" --seq-len 256 \
    --max-tokens 20000000 --val-tokens 1000000 --num-train-files 2
python scripts/prepare_chat.py --out "$data/mid" --seq-len 256 --max-tokens 3000000 \
    --math-frac 0.25 --val-tokens 300000 --limit-per-source 3000
python scripts/prepare_chat.py --out "$data/sft" --seq-len 256 --max-tokens 1000000 \
    --math-frac 0.25 --val-tokens 300000 --limit-per-source 3000 --seed 1
