#!/bin/bash
# Exemplos de geracao (M e SFT do Hugging Face) em CPU. Uso: sbatch slurm/examples.sbatch
#SBATCH -J slm-examples
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --no-requeue
#SBATCH -o logs/%x.%j.out

set -euo pipefail
umask 002
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
H="${SLM_SHARED:-/scratch/ap211/slm-shared}/hf/slm-114m"
python scripts/gen_examples.py --model "M=$H/mid_m/final.pt" --model "SFT=$H/sft/final.pt" --output results/final/exemplos.md
